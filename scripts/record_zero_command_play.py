"""Record a controlled zero-command play rollout from a trained RSL-RL checkpoint.

This script is intended for standing-still checks.  It forces the velocity
command to exactly zero, removes push events, and by default removes
``reset_from_motion`` so the rollout starts from the environment's default
reset pose instead of a random AMP motion frame.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Configure headless rendering before importing mujoco/mjlab modules.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import torch

import src.compat.warp_context_shim  # noqa: F401  # before mjlab (warp-lang>=1.13)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from rsl_rl.runners.amp_on_policy_runner import _unpack_obs


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Record a 0-command video from a trained checkpoint."
  )
  parser.add_argument("task", help="Registered task id, e.g. Unitree-G1-AMP-Flat")
  parser.add_argument(
    "--checkpoint-file",
    required=True,
    type=Path,
    help="Path to model_*.pt checkpoint to load.",
  )
  parser.add_argument(
    "--seconds",
    type=float,
    default=10.0,
    help="Video duration in seconds. Default: 10.0",
  )
  parser.add_argument(
    "--control-dt",
    type=float,
    default=0.02,
    help="Policy/control step in seconds. Default matches AMP env: 0.02 (50 Hz).",
  )
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--device", default=None)
  parser.add_argument("--video-width", type=int, default=1280)
  parser.add_argument("--video-height", type=int, default=720)
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Directory for mp4 output. Default: <checkpoint-run>/videos/zero_command_play_<timestamp>",
  )
  parser.add_argument(
    "--keep-reset-from-motion",
    action="store_true",
    help="Keep AMP reset_from_motion. By default it is removed for pure standing eval.",
  )
  parser.add_argument(
    "--keep-pushes",
    action="store_true",
    help="Keep push_robot interval event. By default it is removed.",
  )
  parser.add_argument(
    "--keep-terminations",
    action="store_true",
    help="Keep terminations. By default they are removed so falls are visible in video.",
  )
  return parser.parse_args()


def force_zero_velocity_command(env_cfg: Any) -> None:
  if "twist" not in env_cfg.commands:
    raise KeyError("Expected a 'twist' velocity command in env_cfg.commands")

  twist = env_cfg.commands["twist"]
  if not isinstance(twist, UniformVelocityCommandCfg):
    raise TypeError(f"Expected UniformVelocityCommandCfg, got {type(twist)!r}")

  twist.rel_standing_envs = 1.0
  twist.rel_heading_envs = 0.0
  twist.heading_command = False
  twist.ranges.lin_vel_x = (0.0, 0.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.ranges.heading = None


def configure_zero_command_env(env_cfg: Any, args: argparse.Namespace) -> None:
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.viewer.width = args.video_width
  env_cfg.viewer.height = args.video_height

  force_zero_velocity_command(env_cfg)

  if not args.keep_pushes:
    env_cfg.events.pop("push_robot", None)
  if not args.keep_reset_from_motion:
    env_cfg.events.pop("reset_from_motion", None)
  if not args.keep_terminations:
    env_cfg.terminations = {}


def output_dir_for(checkpoint: Path, requested: Path | None) -> Path:
  if requested is not None:
    return requested
  stamp = time.strftime("%Y%m%d_%H%M%S")
  return checkpoint.parent / "videos" / f"zero_command_play_{stamp}"


def record_zero_command(args: argparse.Namespace) -> dict[str, Any]:
  checkpoint = args.checkpoint_file.resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
  configure_torch_backends()

  # Import task packages after environment setup so the registry is populated.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  configure_zero_command_env(env_cfg, args)

  video_length = max(1, int(round(args.seconds / args.control_dt)))
  video_dir = output_dir_for(checkpoint, args.output_dir).resolve()
  video_dir.mkdir(parents=True, exist_ok=True)

  base_env = None
  vec_env = None
  try:
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    rec_env = VideoRecorder(
      base_env,
      video_folder=video_dir,
      step_trigger=lambda step: step == 0,
      video_length=video_length,
      disable_logger=True,
    )
    vec_env = RslRlVecEnvWrapper(rec_env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    obs, _ = _unpack_obs(vec_env.get_observations())
    lin_vel_xy: list[float] = []
    yaw_rate: list[float] = []
    roll_pitch_rate: list[float] = []
    heights: list[float] = []

    for _ in range(video_length):
      with torch.no_grad():
        actions = policy(obs)
      obs_dict, _rew, _done, _info = vec_env.step(actions)
      obs, _ = _unpack_obs(obs_dict)

      robot = vec_env.unwrapped.scene["robot"]
      lin_vel_xy.append(robot.data.root_link_lin_vel_b[0, :2].detach().norm().item())
      ang = robot.data.root_link_ang_vel_b[0].detach()
      yaw_rate.append(abs(ang[2].item()))
      roll_pitch_rate.append(float(torch.norm(ang[:2]).item()))
      heights.append(float(robot.data.root_link_pos_w[0, 2].detach().item()))
  finally:
    if vec_env is not None:
      vec_env.close()
    elif base_env is not None:
      base_env.close()

  videos = sorted(video_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
  video = videos[-1] if videos else None
  if video is None:
    raise RuntimeError(f"No mp4 video was written under: {video_dir}")

  return {
    "checkpoint": str(checkpoint),
    "task": args.task,
    "seconds_requested": args.seconds,
    "steps": video_length,
    "video": str(video),
    "video_dir": str(video_dir),
    "mean_lin_vel_xy": sum(lin_vel_xy) / len(lin_vel_xy),
    "max_lin_vel_xy": max(lin_vel_xy),
    "mean_abs_yaw_rate": sum(yaw_rate) / len(yaw_rate),
    "max_abs_yaw_rate": max(yaw_rate),
    "mean_roll_pitch_rate": sum(roll_pitch_rate) / len(roll_pitch_rate),
    "max_roll_pitch_rate": max(roll_pitch_rate),
    "height_start": heights[0],
    "height_end": heights[-1],
    "height_min": min(heights),
    "height_max": max(heights),
  }


def main() -> None:
  result = record_zero_command(parse_args())
  print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
