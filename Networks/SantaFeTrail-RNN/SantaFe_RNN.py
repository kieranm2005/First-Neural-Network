import sys
import os
sys.path.append(os.path.abspath("/u/kieranm/Documents/Python/First-Neural-Network/Environments"))  # Add absolute Environments path to sys.path

from HorizontalLineEnv import SantaFeTrailEnv

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from torch.optim.lr_scheduler import StepLR
import json
import datetime
import os
from tqdm import trange

from rnn_model import SantaFeLSTM
from replay_buffer import PrioritizedReplayBuffer
from utils import save_stats, save_best_transitions, load_best_transitions, save_model, load_model
from config import num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, learning_rate, replay_buffer_size, recent_buffer_size, hidden_size, target_update_freq, n_step

# Registering custom environment
gym.register(
        id="gymnasium_env/HorizontalLine-v0",
        entry_point="HorizontalLineEnv:SantaFeTrailEnv",
        reward_threshold=32,
        max_episode_steps=48
    )

# Initializing environment
env = gym.make("gymnasium_env/HorizontalLine-v0", render_mode="rgb_array")

video_folder = "./videos"
env = RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=lambda episode_id: episode_id % 200 == 0,
    name_prefix="SantaFeLSTM"
)

# Simplify to standard DQN logic
replay_buffer = deque(maxlen=replay_buffer_size)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and optimizer
observation_shape = env.observation_space.shape
if len(observation_shape) != 1:
    raise ValueError(f"Expected observation_space.shape to be 1D (e.g., (input_dim,)), got {observation_shape}")
num_actions = env.action_space.n
model = SantaFeLSTM(observation_shape, hidden_size, num_actions).to(device)
target_model = SantaFeLSTM(observation_shape, hidden_size, num_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epsilon = epsilon_start

# Load previous model if it exists
model_loaded, loaded_stats, epsilon = load_model(model, optimizer, epsilon_start)
if model_loaded:
    episode_stats = loaded_stats
    print("Loaded previous model and training stats")

# Load best transitions if they exist
best_transitions = load_best_transitions()
if best_transitions:
    for transition in best_transitions:
        replay_buffer.append(transition)
    print(f"Loaded {len(best_transitions)} best transitions")

# Track best episodes
best_episodes = []

# Training loop
step_count = 0
episode_stats = []

save_every = 50  # Save stats/model every N episodes

for episode in trange(num_episodes, desc="Training"):  # Progress bar
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = model(obs)
                action = torch.argmax(q_values, dim=1).item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward += reward

        # Store transition in replay buffer
        transition = (obs, action, reward, next_obs_tensor, done)
        replay_buffer.append(transition)
        obs = next_obs_tensor

        # Train the model if replay buffer has enough samples
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

            obs_batch = torch.cat(obs_batch)
            next_obs_batch = torch.cat(next_obs_batch)
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
            done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device)

            # Select the Q-value for each action in the batch using gather
            raw_model_out = model(obs_batch)
            if raw_model_out.dim() == 3:  # (batch_size, seq_len, num_actions)
                model_out = raw_model_out[:, -1, :]  # Use last time step
            else:
                model_out = raw_model_out
            q_values = model_out.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                target_out = target_model(next_obs_batch)
                if target_out.dim() == 3:
                    target_out = target_out[:, -1, :]
                next_q_values = target_out.max(1)[0]
                targets = reward_batch + (1 - done_batch) * gamma * next_q_values

            # Compute loss and update model
            loss = loss_fn(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target model periodically
        step_count += 1
        if step_count % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

    # End of episode
    print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
    episode_stats.append({
        "episode": episode + 1,
        "total_reward": total_reward,
        "epsilon": epsilon
    })

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # Save model and stats periodically
    if (episode + 1) % save_every == 0:
        save_model(model, optimizer, episode_stats, epsilon)
        stats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../Data/SantaFeTrail-RNN')
        os.makedirs(stats_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_path = os.path.join(stats_dir, f"episode_stats_{timestamp}.json")
        with open(stats_path, "w") as f:
            json.dump(episode_stats, f)
        print(f"Saved model and episode stats at episode {episode+1}")

# Clean up video recorder
if hasattr(env, "video_recorder") and env.video_recorder is not None:
    env.video_recorder.close()
    env.video_recorder = None

env.close()

# Sort episodes by reward and save the transitions from the best 200 episodes
best_episodes.sort(key=lambda x: x[0], reverse=True)
best_transitions = []
for _, transitions in best_episodes[:200]:
    best_transitions.extend(transitions)

# Save best transitions and final model
save_best_transitions(best_transitions)
save_model(model, optimizer, episode_stats, epsilon)
print(f"Saved {len(best_transitions)} transitions from best episodes")

# Save episode stats to Data/SantaFeTrail-RNN
stats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../Data/SantaFeTrail-RNN')
os.makedirs(stats_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
stats_path = os.path.join(stats_dir, f"episode_stats_{timestamp}.json")
with open(stats_path, "w") as f:
    json.dump(episode_stats, f)
print(f"Saved episode stats to {stats_path}")
