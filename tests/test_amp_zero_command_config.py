"""Regression tests for AMP zero-command standing configuration."""

import unittest

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_ppo_runner_cfg
from src.tasks.amp_loco import mdp


class AmpZeroCommandConfigTest(unittest.TestCase):
  def test_amp_flat_samples_enough_zero_command_envs(self):
    cfg = g1_amp_flat_env_cfg()
    twist_cmd = cfg.commands["twist"]

    self.assertIsInstance(twist_cmd, UniformVelocityCommandCfg)
    self.assertEqual(twist_cmd.rel_standing_envs, 0.25)

  def test_amp_flat_has_zero_command_standing_rewards(self):
    cfg = g1_amp_flat_env_cfg()

    for name in (
      "stand_still",
      "zero_command_body_velocity_l2",
      "zero_command_foot_slip",
    ):
      self.assertIn(name, cfg.rewards)
      self.assertIsInstance(cfg.rewards[name], RewardTermCfg)

    self.assertIs(cfg.rewards["stand_still"].func, mdp.stand_still)
    self.assertEqual(cfg.rewards["stand_still"].weight, -1.0)
    self.assertIs(
      cfg.rewards["zero_command_body_velocity_l2"].func,
      mdp.zero_command_body_velocity_l2,
    )
    self.assertEqual(cfg.rewards["zero_command_body_velocity_l2"].weight, -2.0)
    self.assertIs(
      cfg.rewards["zero_command_foot_slip"].func,
      mdp.zero_command_feet_slip,
    )
    self.assertEqual(cfg.rewards["zero_command_foot_slip"].weight, -0.5)

  def test_amp_finetune_config_uses_lower_learning_rate_and_more_iterations(self):
    cfg = g1_amp_ppo_runner_cfg()

    self.assertEqual(cfg.algorithm.learning_rate, 5.0e-4)
    self.assertEqual(cfg.max_iterations, 120001)


if __name__ == "__main__":
  unittest.main()
