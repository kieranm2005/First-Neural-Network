import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Define the original trail coordinates. Assuming (x,y) with (0,0) at bottom-left.
original_trail = ([(0, 0), (0, 1), (0, 2), (0, 3),
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
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
    
    def __init__(self, size: int = 32, food_locations=None, render_mode=None):
        super().__init__()
        self.size = size
        self.render_mode = render_mode
        # Internal agent location (x, y) - consistent with Cartesian, (0,0) bottom-left
        self._agent_location = np.array([0, 0]) 
        
        # Directions: 0: right (+x), 1: up (+y), 2: left (-x), 3: down (-y)
        self._directions_map = {
            0: np.array([1, 0]),   # right (+X)
            1: np.array([0, 1]),   # up (+Y)
            2: np.array([-1, 0]),  # left (-X)
            3: np.array([0, -1]),  # down (-Y)
        }
        
        self._agent_direction = 0 

        # Store original trail for transformations at reset
        self.original_food_locations = set(original_trail) 
        self._food_locations = set()

        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.fig = None
        self.ax = None
        self.imshow_obj = None
        self.agent_patch = None

    # Simpler transformation: Direct 90 degrees Counter-Clockwise rotation
    # (x_original, y_original) -> (y_original, (self.size - 1) - x_original)
    def _transform_coords(self, x_orig, y_orig):
        x_final = y_orig
        y_final = (self.size - 1) - x_orig
        return x_final, y_final

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

        if action == 2 and current_loc_tuple in self._food_locations:
            reward = 1
            self._food_locations.discard(current_loc_tuple)

        terminated = len(self._food_locations) == 0
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Apply transformation to initial agent location
        initial_x_agent = 0
        initial_y_agent = 0 
        transformed_x, transformed_y = self._transform_coords(initial_x_agent, initial_y_agent)
        self._agent_location = np.array([transformed_x, transformed_y])
        
        # Determine initial agent direction in the transformed space
        # Original 0 (Right: +X) -> New 1 (Up: +Y) after 90 CCW rotation of environment
        # Original 1 (Up: +Y) -> New 2 (Left: -X)
        # Original 2 (Left: -X) -> New 3 (Down: -Y)
        # Original 3 (Down: -Y) -> New 0 (Right: +X)
        self._agent_direction = 1 # Start facing Up in the transformed world

        # Apply transformation to all food locations
        self._food_locations = set()
        for fx, fy in self.original_food_locations:
            transformed_fx, transformed_fy = self._transform_coords(fx, fy)
            self._food_locations.add((transformed_fx, transformed_fy))
            
        return self._get_obs(), {}
    
    def render(self, mode="human"):
        if self.render_mode == "rgb_array" or mode == "rgb_array":
            # Return an RGB array of the current state
            # Example: return np.zeros((height, width, 3), dtype=np.uint8)
            return self._get_rgb_array()
        elif self.render_mode == "human" or mode == "human":
            # Optionally, display to screen
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
        id="gymnasium_env/SantaFeTrail-v0",
        entry_point="santa_fe_env:SantaFeTrailEnv", # Assuming saved as santa_fe_env.py
        reward_threshold=89, # This might need adjustment if transformed trail makes it harder/easier
        max_episode_steps=400 # This might need adjustment
    )

if __name__ == '__main__':
    register()
    env = gym.make("gymnasium_env/SantaFeTrail-v0")

    obs, info = env.reset()
    # Original (0,0) (bottom-left) -> Final transformed (0, 31) (top-left) if size=32
    print("Initial agent location:", env._agent_location) 
    print("Initial agent direction (0:right, 1:up, 2:left, 3:down):", env._agent_direction)

    num_steps = 200
    print("\n--- Testing actions ---")
    for i in range(num_steps):
        # Example sequence to move agent to test transformed environment
        # Agent starts at (0, 31) (top-left) and faces Up (1)
        if i < 5: action = 2 # Move forward (agent starts facing up, so it moves up)
        elif i < 7: action = 0 # Turn left (to face Left)
        elif i < 12: action = 2 # Move forward (left)
        elif i < 14: action = 0 # Turn left (to face Down)
        elif i < 19: action = 2 # Move forward (down)
        elif i < 21: action = 0 # Turn left (to face Right)
        elif i < 26: action = 2 # Move forward (right)
        else: action = 1 # Keep turning right for demonstration
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print(f"Episode finished at step {i+1}. Food remaining: {len(env._food_locations)}")
            obs, info = env.reset()
            print("Environment reset.")
            
    env.close()

    print("\n--- Manual initial state verification ---")
    env_test = SantaFeTrailEnv()
    env_test.reset()
    env_test.render()
    plt.title("Manual Transformed Initial State (Close to proceed)")
    plt.show(block=True) 
    env_test.close()