"""Extract or tile a low-motion clip into a 5-10s standing reference NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_ARRAY_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _best_window(joint_vel: np.ndarray, window: int) -> tuple[int, int]:
    speed = np.linalg.norm(joint_vel, axis=1)
    best_start = 0
    best_score = float("inf")
    for start in range(0, joint_vel.shape[0] - window + 1):
        score = float(speed[start : start + window].mean())
        if score < best_score:
            best_score = score
            best_start = start
    return best_start, best_start + window


def _tile_array(array: np.ndarray, target_len: int) -> np.ndarray:
    reps = int(np.ceil(target_len / array.shape[0]))
    tiled = np.tile(array, (reps,) + (1,) * (array.ndim - 1))
    return tiled[:target_len]


def make_standing_npz(
    source: Path,
    output: Path,
    target_duration_s: float = 7.5,
    min_duration_s: float = 5.0,
    max_duration_s: float = 10.0,
) -> None:
    data = np.load(source)
    fps = float(data["fps"][0])
    target_duration_s = float(np.clip(target_duration_s, min_duration_s, max_duration_s))
    target_frames = max(1, int(round(target_duration_s * fps)))

    total_frames = data["joint_pos"].shape[0]
    window = min(total_frames, max(1, int(round(min(target_duration_s, total_frames / fps) * fps))))

    start, end = _best_window(data["joint_vel"], window)
    segment = {key: data[key][start:end] for key in _ARRAY_KEYS}

    clip_len = segment["joint_pos"].shape[0]
    if clip_len >= target_frames:
        sliced = {key: value[:target_frames] for key, value in segment.items()}
    else:
        sliced = {key: _tile_array(value, target_frames) for key, value in segment.items()}

    # Standing reference should be quasi-static.
    sliced["joint_vel"][:] = 0.0
    sliced["body_lin_vel_w"][:] = 0.0
    sliced["body_ang_vel_w"][:] = 0.0

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, fps=np.array([fps], dtype=np.float64), **sliced)

    duration_s = sliced["joint_pos"].shape[0] / fps
    print(
        f"Wrote {output} | fps={fps:.0f} frames={sliced['joint_pos'].shape[0]} "
        f"duration={duration_s:.2f}s | source_window=[{start}:{end}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("src/assets/motions/g1/amp/WalkandRun/step_rotate_idle_000_002__A026.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/assets/motions/g1/amp/Standing/stand_idle_7s5.npz"),
    )
    parser.add_argument("--target-duration-s", type=float, default=7.5)
    args = parser.parse_args()
    make_standing_npz(args.source, args.output, target_duration_s=args.target_duration_s)


if __name__ == "__main__":
    main()
