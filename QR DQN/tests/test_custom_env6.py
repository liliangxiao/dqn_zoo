# tests/test_custom_env6.py

import sys
import os
import gymnasium as gym
import time

# Adjust the path to ensure Python can find the custom_env module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from custom_env.custom_env6 import CustomEnv6  # Import the CustomEnv6 class

def main():
    # Define start and goal positions
    start = [34,0]  # Replace with actual start coordinates
    goal = [118,198]    # Replace with actual goal coordinates
    # Path to your .mat data file
    data_file_path = os.path.join(parent_dir, 'Toronto_map_complex_all.mat')

    # Check if the data file exists
    if not os.path.exists(data_file_path):
        print(f"Data file not found at path: {data_file_path}")
        sys.exit(1)

    # Create environment
    env = CustomEnv6(start, goal, data_file=data_file_path, render_mode='human')

    try:
        # Reset environment
        obs, info = env.reset()

        done = False
        total_reward = 0

        while not done:
            # Render the environment
            env.render()

            # Sample random action (replace with your agent's action logic)
            action = env.action_space.sample()

            # Take action
            obs, reward, done, info = env.step(action)

            total_reward += reward
            print(f"Action: {action}, Reward: {reward}, Done: {done}")

            # Control simulation speed
            time.sleep(0.05)  # 20 FPS

        print(f"Total Reward: {total_reward}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
