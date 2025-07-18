import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import sys
import TrailReader as tr

# Define the original trail coordinates. Assuming (x,y) with (0,0) at bottom-left.
original_trail = (tr.load_trail_coordinates('/u/kieranm/Documents/Python/First-Neural-Network/Environments/Trails/SantaFe_coordinates.txt'))
agent_location = np.array(original_trail[0])  # Agent starts at the first loaded coordinate
class SantaFeTrailEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
    
    def __init__(self, size: int = 32, food_locations=None, render_mode=None):
        super().__init__()
        self.size = size
        self.render_mode = render_mode
        # Internal agent location (x, y) - consistent with Cartesian, (0,0) bottom-left
        self._agent_location = agent_location.copy()  # Start at (0, 0)

        # Directions: 0: right (+x), 1: up (+y), 2: left (-x), 3: down (-y)
        self._directions_map = {
            0: np.array([1, 0]),   # right (+X)
            1: np.array([0, 1]),   # up (+Y)
            2: np.array([-1, 0]),  # left (-X)
            3: np.array([0, -1]),  # down (-Y)
        }
        
        self._agent_direction = 0 

        # Store original trail for reset
        self.original_food_locations = set(original_trail) 
        self._food_locations = set()

        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.fig = None
        self.ax = None
        self.imshow_obj = None
        self.agent_patch = None

    def _get_obs(self):
        dir_vec = self._directions_map[self._agent_direction]
        front = self._agent_location + dir_vec
        fx, fy = front
        if (0 <= fx < self.size) and (0 <= fy < self.size):
            has_food = 1.0 if (fx, fy) in self._food_locations else 0.0
        else:
            has_food = 0.0
        return np.array([has_food], dtype=np.float32)

    def step(self, action):
        if action == 0:  # turn left (counter-clockwise)
            self._agent_direction = (self._agent_direction + 1) % 4
        elif action == 1:  # turn right (clockwise)
            self._agent_direction = (self._agent_direction - 1 + 4) % 4
        elif action == 2:  # move forward
            move = self._directions_map[self._agent_direction]
            new_loc = self._agent_location + move
            if (0 <= new_loc[0] < self.size) and (0 <= new_loc[1] < self.size):
                self._agent_location = new_loc

        reward = 0
        current_loc_tuple = tuple(self._agent_location)

        # Reward if agent is on a food square (either by moving or by turning)
        if current_loc_tuple in self._food_locations:
            reward = 1
            self._food_locations.discard(current_loc_tuple)

        terminated = len(self._food_locations) == 0
        truncated = False

        obs = self._get_obs()
        info = {}
        info['position'] = self._agent_location # Track agent's coordinates
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # No transformation: agent starts at (0, 0), facing right (+X)
        self._agent_location = agent_location.copy()
        self._agent_direction = 0 # Right (+X)

        # No transformation: food locations as in original trail
        self._food_locations = set(self.original_food_locations)

        # Reward if agent starts on a food square
        current_loc_tuple = tuple(self._agent_location)
        self.starting_reward = 0
        if current_loc_tuple in self._food_locations:
            self.starting_reward = 1
            self._food_locations.discard(current_loc_tuple)
            
        obs = self._get_obs()
        info = {}
        info['position'] = self._agent_location  # or whatever variable tracks the agent's coordinates
        return obs, info
    
    def render(self, mode="human"):
        if self.render_mode == "rgb_array" or mode == "rgb_array":
            return self._get_rgb_array()
        elif self.render_mode == "human" or mode == "human":
            pass

    def _get_rgb_array(self):
        # Create a blank white image
        img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255

        # Draw food as green
        for fx, fy in self._food_locations:
            if 0 <= fx < self.size and 0 <= fy < self.size:
                img[fy, fx] = [0, 200, 0]

        # Draw agent as red
        ax, ay = self._agent_location
        if 0 <= ax < self.size and 0 <= ay < self.size:
            img[ay, ax] = [200, 0, 0]

            # Draw agent direction as a blue pixel in front of agent
            dir_vec = self._directions_map[self._agent_direction]
            front = self._agent_location + dir_vec
            fx, fy = front
            if 0 <= fx < self.size and 0 <= fy < self.size:
                img[fy, fx] = [0, 0, 200]

        # Upscale for visibility (optional, e.g. 10x)
        scale = 10
        img = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))

        return img
        
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.imshow_obj = None
            self.agent_patch = None

def register():
    gym.register(
        id="gymnasium_env/HorizontalLineEnv-v0",
        entry_point="HorizontalLineEnv:HorizontalLineEnv",
        reward_threshold=32,
        max_episode_steps=48 # Adjusted for horizontal line length
    )

