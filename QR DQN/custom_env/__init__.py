# custom_env/__init__.py

from gymnasium.envs.registration import register
from custom_env.custom_env6 import CustomEnv6  # Absolute import

register(
    id='CustomEnv6-v0',
    entry_point='custom_env.custom_env6:CustomEnv6',
    max_episode_steps=1000,
    reward_threshold=10000,
)

__all__ = ['CustomEnv6']  # Optional: Defines what is exported when using 'from custom_env import *'
