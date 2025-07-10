import sys
import os
sys.path.append(os.path.abspath("../Environments"))
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

replay_buffer = deque(maxlen=replay_buffer_size)
recent_buffer = deque(maxlen=recent_buffer_size)

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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

epsilon = epsilon_start
prioritized_replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_size)

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
        prioritized_replay_buffer.add(*transition)
    print(f"Loaded {len(best_transitions)} best transitions")

# Track best episodes
best_episodes = []

# Training loop
step_count = 0
episode_stats = []
reward_threshold = 1  # Initial value

n_step = 4
n_step_buffer = deque(maxlen=n_step)
for episode in range(num_episodes):
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    done = False
    total_reward = 0
    episode_transitions = []

    episode_transitions = []
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
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        total_reward += reward

        # Store transition in replay buffer
        transition = (obs, action, reward, next_obs_tensor, done)
        replay_buffer.append(transition)
        recent_buffer.append(transition)
        prioritized_replay_buffer.add(*transition)
        episode_transitions.append(transition)
        obs = next_obs_tensor

        # Store in n-step buffer
        n_step_buffer.append((obs, action, reward, next_obs_tensor, done))

        # N-step bootstrapping and training
        if len(n_step_buffer) == n_step and len(prioritized_replay_buffer.buffer) >= batch_size:
            # Calculate n-step return
            R = 0
            for i in range(n_step):
                R += n_step_buffer[i][2] * (gamma ** i)
                if n_step_buffer[i][4]:  # if done
                    break
            
            state, action, _, _, _ = n_step_buffer[0]
            _, _, _, next_state, done = n_step_buffer[-1]
            
            # Store n-step transition
            replay_buffer.append((state, action, R, next_state, done))
            
            # Sample from prioritized replay buffer
            batch, indices, weights = prioritized_replay_buffer.sample(batch_size)
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

            # Process batches
            obs_batch = torch.stack([
                torch.tensor(o, dtype=torch.float32).to(device).squeeze(0)
                if isinstance(o, np.ndarray) else o.to(device).squeeze(0)
                for o in obs_batch
            ])
            next_obs_batch = torch.stack([
                torch.tensor(o, dtype=torch.float32).to(device).squeeze(0)
                if isinstance(o, np.ndarray) else o.to(device).squeeze(0)
                for o in next_obs_batch
            ])
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
            done_batch = torch.tensor([float(d) for d in done_batch], dtype=torch.float32, device=device)

            # Ensure batches are on the correct device
            obs_batch = obs_batch.to(device)
            next_obs_batch = next_obs_batch.to(device)

            # CrossQ update
            q_logits = model(obs_batch)  # shape: [batch, num_actions]
            with torch.no_grad():
                next_q_values = target_model(next_obs_batch)
                target_actions = next_q_values.argmax(dim=1)  # shape: [batch]
            target_dist = torch.zeros_like(q_logits)
            target_dist[range(q_logits.size(0)), target_actions] = 1.0
            log_probs = torch.log_softmax(q_logits, dim=1)
            loss_per_sample = -(target_dist * log_probs).sum(dim=1)
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            loss = (loss_per_sample * weights_tensor).mean()

            # Update network
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update priorities (use loss as proxy for TD error)
            new_priorities = loss_per_sample.detach().cpu().numpy() + 1e-6
            prioritized_replay_buffer.update_priorities(indices, new_priorities)

        step_count += 1
        if step_count % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
    episode_stats.append({
        "episode": episode + 1,
        "total_reward": total_reward,
        "epsilon": epsilon
    })
    # Use average reward over last 10 episodes as metric
    avg_reward = np.mean([stat["total_reward"] for stat in episode_stats[-10:]])
    scheduler.step(avg_reward)

    # Dynamically set reward_threshold to 75th percentile of all episode rewards so far
    if len(episode_stats) >= 4:  # Only update if enough episodes
        rewards = [stat["total_reward"] for stat in episode_stats]
        reward_threshold = np.percentile(rewards, 75)

    # After the episode ends, filter by reward
    if total_reward >= reward_threshold:
        for transition in episode_transitions:
            replay_buffer.append(transition)
            prioritized_replay_buffer.add(*transition)
    # Optionally, store all transitions in recent_buffer for analysis

    # Store episode if it's among the best
    best_episodes.append((total_reward, episode_transitions))
    best_episodes.sort(key=lambda x: x[0], reverse=True)
    if len(best_episodes) > 200:
        best_episodes.pop()

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
