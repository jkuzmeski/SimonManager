#!/usr/bin/env python

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to test Simon Half velocity task configuration.
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test Simon Half velocity task.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-SimonHalf-v0", help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

import isaaclab_tasks.manager_based.locomotion.velocity.config.simon_half  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv


def main():
    """Main function."""
    # get the environment configuration
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    # parse the configuration
    env_cfg_cls = env_cfg.split(":")[1]
    env_cfg_module = env_cfg.split(":")[0]
    
    # import the module
    import importlib
    cfg_module = importlib.import_module(env_cfg_module)
    cfg_class = getattr(cfg_module, env_cfg_cls)
    
    # create environment configuration
    env_cfg = cfg_class()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # check if environment is ManagerBasedRLEnv
    if isinstance(env.unwrapped, ManagerBasedRLEnv):
        print(f"[INFO]: Created environment: {args_cli.task}")
        print(f"[INFO]: Environment name: {env.unwrapped.cfg.scene}")
        print(f"[INFO]: Number of environments: {env.unwrapped.num_envs}")
        print(f"[INFO]: Environment device: {env.unwrapped.device}")
        print(f"[INFO]: Observation space: {env.observation_space}")
        print(f"[INFO]: Action space: {env.action_space}")
    
    # reset environment
    obs, _ = env.reset()
    print(f"[INFO]: Observation shape: {obs.shape}")
    
    # run a few steps
    for i in range(10):
        # sample random actions
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)
        print(f"Step {i}: Reward mean: {reward.mean():.3f}")
        
        if terminated.any() or truncated.any():
            print(f"[INFO]: Some environments terminated or truncated at step {i}")
    
    # close environment
    env.close()
    print("[INFO]: Environment test completed successfully!")


if __name__ == "__main__":
    main()
    simulation_app.close()
