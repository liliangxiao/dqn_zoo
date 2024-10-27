# custom_env/custom_env6.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import time
from scipy.io import loadmat
import os

class CustomEnv6(gym.Env):
    """
    Custom Environment that follows Gymnasium interface.
    This environment simulates a vehicle navigating through a map with traffic lights and stop signs.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, start, goal, data_file='Toronto_map_complex_all.mat', render_mode=None):
        super(CustomEnv6, self).__init__()

        # Determine the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the full path to the .mat file
        data_file_path = os.path.join(current_dir, data_file)

        # Load data from .mat file
        data = loadmat(data_file_path)

        # Inspect available variables
        print("Loaded .mat file variables:", data.keys())

        # Extract variables (ensure these keys exist in your .mat file)
        try:
            self.Map = data['map']
            self.StopSigns = data['stopSigns'].T  # Shape: (N, 2)
            self.trafficLights = data['trafficLights'].T  # Shape: (M, 2)
            self.SpeedMap = data['speedLimits']
        except KeyError as e:
            raise KeyError(f"Missing expected variable in .mat file: {e}")

        # Environment parameters
        self.StopSignReward = 5
        self.trafficLightReward = 15
        self.StartState = np.array(start, dtype=np.int32)
        self.GoalState = np.array(goal, dtype=np.int32)
        self.CurrentState = np.array(start, dtype=np.int32)
        self.OldState = np.array(start, dtype=np.int32)
        self.Bridges = np.array([70, 134])  # Adjust as per actual bridge positions

        # Validate start and goal positions
        self.Road = 1
        self.obstacle = 0

        if self.Map[self.StartState[0], self.StartState[1]] != self.Road:
            raise ValueError("Start position is not on a road.")
        if self.Map[self.GoalState[0], self.GoalState[1]] != self.Road:
            raise ValueError("Goal position is not on a road.")

        # Rewards
        self.GoalReward = 10000
        self.CollisionReward = -5

        # Action space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        # Observation space: [current_x, current_y, goal_x, goal_y]
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([
            self.Map.shape[0],
            self.Map.shape[1],
            self.Map.shape[0],
            self.Map.shape[1]
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.cell_size = 5  # Size of each cell in pixels
        self.margin = 20  # Margin around the map
        self.colors = {
            'road': (255, 255, 255),       # White
            'obstacle': (0, 0, 0),         # Black
            'stop_sign': (255, 0, 0),      # Red
            'traffic_light': (255, 255, 0), # Yellow
            'goal': (0, 0, 255),           # Blue
            'start': (0, 255, 0),          # Green
            'vehicle': (0, 255, 255)       # Cyan for vehicle
        }


        if self.render_mode == 'human':
            pygame.init()
            pygame.display.set_caption("CustomEnv6 Simulation")
            map_width, map_height = self.Map.shape[1], self.Map.shape[0]
            self.screen_width = map_width * self.cell_size + 2 * self.margin
            self.screen_height = map_height * self.cell_size + 2 * self.margin
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self._render_new_map()

    def step(self, action):
        done = False
        reward = 0

        # Handle Pygame events
        if self.render_mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        # Compute next position based on action
        next_position = self.CurrentState.copy()
        if action == 0:  # Up
            next_position[0] -= 1
        elif action == 1:  # Down
            next_position[0] += 1
        elif action == 2:  # Left
            next_position[1] -= 1
        elif action == 3:  # Right
            next_position[1] += 1
        else:
            raise ValueError("Invalid action.")

        # Check boundaries
        is_outside = (
            next_position[0] < 0 or next_position[0] >= self.Map.shape[0] or
            next_position[1] < 0 or next_position[1] >= self.Map.shape[1]
        )

        # Check collision
        collision = False
        if is_outside or self.Map[next_position[0], next_position[1]] == self.obstacle:
            collision = True
            reward += self.CollisionReward
            next_position = self.OldState.copy()  # Step back

        # Update state
        self.OldState = self.CurrentState.copy()
        self.CurrentState = next_position.copy()

        # Rendering
        if self.render_mode == 'human':
            self._render_vehicle()

        # Observation
        observation = np.concatenate((self.CurrentState, self.GoalState)).astype(np.float32)

        # Check if goal is reached
        if np.linalg.norm(self.CurrentState - self.GoalState) < 1:
            done = True
            reward += self.GoalReward
            return observation, reward, done, {}

        if collision:
            return observation, reward, done, {}

        # Calculate on-road reward
        reward += self._on_road_reward(next_position)

        return observation, reward, done, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.CurrentState = self.StartState.copy()
        self.OldState = self.StartState.copy()
        observation = np.concatenate((self.CurrentState, self.GoalState)).astype(np.float32)

        if self.render_mode == 'human':
            self._render_new_map()
            self._render_vehicle()

        return observation, {}

    def render(self, mode='human'):
        if self.render_mode != 'human':
            return

        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.render_mode == 'human':
            pygame.quit()

    def _on_road_reward(self, next_position):
        reward = 0

        # Reward for distance improvement
        old_distance = np.linalg.norm(self.OldState - self.GoalState)
        new_distance = np.linalg.norm(next_position - self.GoalState)
        if new_distance < old_distance:
            reward += 1 + (
                0.5 * np.linalg.norm(self.CurrentState - self.StartState) -
                0.5 * new_distance
            ) / np.linalg.norm(self.StartState - self.GoalState)
        else:
            reward -= 1

        # Penalty for stop signs
        # Ensure self.StopSigns is reshaped appropriately
        if self.StopSigns.shape[0] == 2 and self.StopSigns.shape[1] > 1:
            # Convert to a shape that allows direct comparison with next_position
            stop_sign_positions = self.StopSigns.T  # Transpose to (4, 2) if each column is a separate stop sign
            # Check if next_position matches any stop sign
            if np.any(np.all(stop_sign_positions == next_position, axis=1)):
                speed = self.SpeedMap[next_position[0], next_position[1]]
                reward -= self.StopSignReward * speed * 10 / 36 / 20



        # Penalty for traffic lights
        if self.trafficLights.shape[0] == 2 and self.trafficLights.shape[1] > 1:
            # Transpose the traffic lights to have coordinates in rows
            traffic_light_positions = self.trafficLights.T  # This should give a shape of (N, 2)

            # Check if next_position matches any traffic light
            if np.any(np.all(traffic_light_positions == next_position, axis=1)):
                speed = self.SpeedMap[next_position[0], next_position[1]]
                reward -= self.trafficLightReward * speed * 10 / 36 / 20


        # Reward for high speed limit
        reward += self.SpeedMap[next_position[0], next_position[1]] / 100

        # Encourage fewer steps
        reward -= 0.1

        return reward

    def _render_new_map(self):
        if self.render_mode != 'human':
            return

        self.screen.fill((0, 0, 0))  # Clear screen with black

        for x in range(self.Map.shape[0]):
            for y in range(self.Map.shape[1]):
                rect = pygame.Rect(
                    self.margin + y * self.cell_size,
                    self.margin + x * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                if self.Map[x, y] == self.Road:
                    color = self.colors['road']
                else:
                    color = self.colors['obstacle']
                pygame.draw.rect(self.screen, color, rect)

        # Draw Stop Signs
        for sign in self.StopSigns.T:
            pos = (
                self.margin + sign[1] * self.cell_size + self.cell_size // 2,
                self.margin + sign[0] * self.cell_size + self.cell_size // 2
            )
            pygame.draw.circle(self.screen, self.colors['stop_sign'], pos, self.cell_size)

        # Draw Traffic Lights
        for light in self.trafficLights.T:
            pos = (
                self.margin + light[1] * self.cell_size + self.cell_size // 2,
                self.margin + light[0] * self.cell_size + self.cell_size // 2
            )
            pygame.draw.circle(self.screen, self.colors['traffic_light'], pos, self.cell_size)

        # Draw Start State
        start_pos = (
            self.margin + self.StartState[1] * self.cell_size + self.cell_size // 2,
            self.margin + self.StartState[0] * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(self.screen, self.colors['start'], start_pos, self.cell_size // 2)

        # Draw Goal State
        goal_pos = (
            self.margin + self.GoalState[1] * self.cell_size + self.cell_size // 2,
            self.margin + self.GoalState[0] * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(self.screen, self.colors['goal'], goal_pos, self.cell_size // 2)

    def _render_vehicle(self):
        if self.render_mode != 'human':
            return

        # Draw the previous position in white if it differs from the current
        if not np.array_equal(self.OldState, self.CurrentState):
            prev_pos = (self.margin + self.OldState[1] * self.cell_size + self.cell_size // 2,
                        self.margin + self.OldState[0] * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(self.screen, (255, 255, 255), prev_pos, self.cell_size)  # White dot for previous position

        # Draw the current position of the vehicle
        vehicle_pos = (self.margin + self.CurrentState[1] * self.cell_size + self.cell_size // 2,
                    self.margin + self.CurrentState[0] * self.cell_size + self.cell_size // 2)
        
        pygame.draw.circle(self.screen, self.colors['vehicle'], vehicle_pos, self.cell_size)

        # Update the display
        pygame.display.flip()  # Refresh the screen


        # Optionally, clear the previous position of the vehicle or trace only the last position
        # Use the last known position for the path if needed
        # Otherwise, just keep the vehicle's current representation


    def optimized_episode(self, display_flag=False):
        """
        Placeholder for optimized episode using A*.
        Implement A* pathfinding if needed.
        """
        raise NotImplementedError("A* optimized episode not implemented.")
