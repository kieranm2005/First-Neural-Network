import sys
import os
sys.path.append(os.path.abspath("../Environments"))
sys.path.append(os.path.abspath("/u/kieranm/Documents/Python/First-Neural-Network/Environments"))  # Add current directory to sys.path

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from HorizontalLineEnv import SantaFeTrailEnv
from gymnasium.wrappers import RecordVideo
import datetime
'''To Do:
1. Vectorized Environment: Use `gym.vector.make` for parallel environments.
2. Collect statistics: Use `env.get_episode_stats()` to collect statistics.
3. Plotting: Use `matplotlib` to visualize the training progress.
4. Tune hyperparameters with Optuna'''

class SantaFeCNN(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(SantaFeCNN, self).__init__()
        # The input is a single value indicating if there's food in front
        self.fc1 = nn.Linear(observation_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Registering custom environment
gym.register(
    id="gymnasium_env/HorizontalLine-v0",
    entry_point="HorizontalLineEnv:SantaFeTrailEnv",
    reward_threshold=22,
    max_episode_steps=100
)

# Initializing environment (NO video during training)
env = gym.make("gymnasium_env/HorizontalLine-v0")

# Hyperparameters
num_episodes = 100
batch_size = 64
gamma = 0.9006346496123904 # Via optuna
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.912 # Via optuna
learning_rate = 1e-3
replay_buffer = deque(maxlen=50000)
recent_buffer = deque(maxlen=5000)  # smaller buffer for recent transitions

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and optimizer
observation_shape = env.observation_space.shape  # (channels, height, width)
num_actions = env.action_space.n
model = SantaFeCNN(observation_shape, num_actions).to(device)
target_model = SantaFeCNN(observation_shape, num_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()
target_update_freq = 1000  # steps

# Define loss function and optimizer
loss_fn = nn.MSELoss() # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

epsilon = epsilon_start

episode_stats = []

step_count = 0
for episode in range(num_episodes):
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)  # shape: (1,)
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = model(obs)
                action = torch.argmax(q_values, dim=0).item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device)
        total_reward += reward

        # Store transition in replay buffer
        replay_buffer.append((obs, action, reward, next_obs_tensor, done))
        recent_buffer.append((obs, action, reward, next_obs_tensor, done))

        obs = next_obs_tensor

        # Sample random minibatch and train
        if len(replay_buffer) >= batch_size:
            if len(recent_buffer) >= batch_size // 2:
                batch = random.sample(replay_buffer, batch_size // 2) + random.sample(recent_buffer, batch_size // 2)
            else:
                batch = random.sample(replay_buffer, batch_size)

            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
            obs_batch = torch.stack([o.to(device) for o in obs_batch])
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
            next_obs_batch = torch.stack([no.to(device) for no in next_obs_batch])
            # Ensure done_batch is 1.0 for terminal, 0.0 for non-terminal
            done_batch = torch.tensor([float(d) for d in done_batch], dtype=torch.float32, device=device)
            # Q(s, a)
            q_values = model(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
            # max_a' Q(s', a')
            with torch.no_grad():
                next_q_values = target_model(next_obs_batch).max(1)[0]
            target = reward_batch + gamma * next_q_values * (1 - done_batch)

            loss = loss_fn(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        step_count += 1
        if step_count % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        # Decay epsilon per step for more granular control
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
    episode_stats.append({
        "episode": episode + 1,
        "total_reward": total_reward,
        "epsilon": epsilon
    })
env.close()

def save_stats(stats):
    import json, datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_stats_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(stats, f)

save_stats(episode_stats)

# --- Render and save video only if last episode's reward > 0 ---
video_dir = "videos"  # Define the directory to save videos
last_reward = episode_stats[-1]["total_reward"]
if last_reward > 0:
    # Add timestamp and reward to video filename prefix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_prefix = f"episode_{timestamp}_reward_{int(last_reward)}"

    env = RecordVideo(
        gym.make("gymnasium_env/HorizontalLine-v0", render_mode="rgb_array"),
        video_folder=video_dir,
        name_prefix=video_prefix,
        episode_trigger=lambda _: True  # Always record (only one episode is run here)
    )
    obs, info = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    done = False
    while not done:
        with torch.no_grad():
            q_values = model(obs)
            action = torch.argmax(q_values, dim=0).item()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    env.close()
    print(f"Video saved to {video_dir} with prefix '{video_prefix}'")
else:
    print("No video saved: last episode reward did not exceed 0.")