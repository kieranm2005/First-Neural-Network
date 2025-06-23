import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

trail = ([(0, 0), (0, 1), (0, 2), (0, 3),
                            (1, 3), (2, 3), (3, 3), (4, 3), (5, 3),
                            (5, 4), (5, 5), (5, 6), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12),
                            (6, 12), (7, 12), (8, 12), (9, 12), (10, 12),
                            (12, 12), (13, 12), (14, 12), (15, 12),
                            (18, 12), (19, 12), (20, 12), (21, 12), (22, 12), (23, 12),
                            (24, 11), (24, 10), (24, 9), (24, 8), (24, 7), (24, 4), (24, 3),
                            (25, 1), (26, 1), (27, 1), (28, 1),
                            (30, 2), (30, 3), (30, 4), (30, 5),
                            (29, 7), (28, 7),
                            (27, 8), (27, 9), (27, 10), (27, 11), (27, 12), (27, 13), (27, 14),
                            (26, 16), (25, 16), (24, 16), (21, 16), (19, 16), (18, 16), (17, 16), (16, 17),
                            (15, 20), (14, 20), (11, 20), (10, 20), (9, 20), (8, 20),
                            (5, 21), (5, 22), (4, 24), (3, 24), (2, 25), (2, 26), (2, 27), (3, 29), (4, 29), (6, 29), (9, 29), (12, 29),
                            (14, 28), (14, 27), (14, 26), (15, 23), (18, 24), (19, 27), (22, 26), (23, 23)]) 

class SantaFeTrailEnv(gym.Env):
    def __init__(self, size: int = 32, food_locations=None):
        self.size = size
        self._agent_location = np.array([0, 0])
        self._agent_direction = 0  # 0: right, 1: down, 2: left, 3: up

        # Food locations: set of (x, y) tuples
        if food_locations is None:
            self._food_locations = set(trail) 
        else:
            self._food_locations = set(food_locations)

        # Observation: [is_food_ahead (0/1)]
        self.observation_space = spaces.Box(0, 1, shape=(1,), dtype=np.int8)

        # Actions: 0 = turn left, 1 = turn right, 2 = move forward
        self.action_space = spaces.Discrete(3)

        # Directions: right, down, left, up
        self._directions = [
            np.array([1, 0]),   # right
            np.array([0, 1]),   # down
            np.array([-1, 0]),  # left
            np.array([0, -1]),  # up
        ]

    def _get_obs(self):
        # Compute the square in front of the agent
        facing = self._directions[self._agent_direction]
        ahead = self._agent_location + facing
        # Check bounds
        if (0 <= ahead[0] < self.size) and (0 <= ahead[1] < self.size):
            is_food_ahead = int(tuple(ahead) in self._food_locations)
        else:
            is_food_ahead = 0 # no food if trying to look outside of grid bounds
        return np.array([is_food_ahead], dtype=np.int8)

    def step(self, action):
        if action == 0:  # turn left
            self._agent_direction = (self._agent_direction - 1) % 4
        elif action == 1:  # turn right
            self._agent_direction = (self._agent_direction + 1) % 4
        elif action == 2:  # move forward
            move = self._directions[self._agent_direction]
            new_loc = self._agent_location + move
            if (0 <= new_loc[0] < self.size) and (0 <= new_loc[1] < self.size):
                self._agent_location = new_loc
                # Eat food if present
                self._food_locations.discard(tuple(self._agent_location))

        # Example reward: +1 for eating food, 0 otherwise
        reward = 1 if tuple(self._agent_location) in self._food_locations else 0

        # Done if all food eaten or max steps reached
        terminated = len(self._food_locations) == 0
        truncated = False  # Or set to True if you have a max step limit

        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        self._agent_location = np.array([0, 0])
        self._agent_direction = 0
        # Reset food locations
        self._food_locations = set(trail) 
        return self._get_obs(), {}
    
    def render(self, mode="human"):
        grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)  # RGB image

        # Draw food as green squares
        for fx, fy in self._food_locations:
            grid[fx, fy] = [0, 255, 0]

        plt.clf()
        plt.imshow(grid, interpolation='none')
        plt.title("Santa Fe Trail")
        plt.axis('off')


        # Draw agent as a triangle
        x, y = self._agent_location
        direction = self._agent_direction

        # Triangle size and orientation
        triangle_size = 0.5
        cx, cy = y, x  # Note: matplotlib uses (col, row) = (y, x)

        # Define triangle vertices for each direction
        if direction == 0:  # right
            verts = [
                (cx + 0.5, cy + 0.25),
                (cx + 0.5, cy + 0.75),
                (cx + 1.0, cy + 0.5)
            ]
        elif direction == 1:  # down
            verts = [
                (cx + 0.25, cy + 0.5),
                (cx + 0.75, cy + 0.5),
                (cx + 0.5, cy + 1.0)
            ]
        elif direction == 2:  # left
            verts = [
                (cx + 0.5, cy + 0.25),
                (cx + 0.5, cy + 0.75),
                (cx, cy + 0.5)
            ]
        elif direction == 3:  # up
            verts = [
                (cx + 0.25, cy + 0.5),
                (cx + 0.75, cy + 0.5),
                (cx + 0.5, cy)
            ]

        triangle = Polygon(verts, closed=True, color='red')
        plt.gca().add_patch(triangle)
        plt.pause(0.1)
    
def register():

    gym.register(
        id="gymnasium_env/SantaFeTrail-v0",
        entry_point="SantaFeTrailEnv:SantaFeTrailEnv",
        reward_threshold=89,
        max_episode_steps=400
        )
