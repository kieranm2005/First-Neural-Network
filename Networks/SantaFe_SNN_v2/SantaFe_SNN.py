import sys
import os
sys.path.append(os.path.abspath("/u/kieranm/Documents/Python/First-Neural-Network/Environments"))
sys.path.append(os.path.abspath("/u/kieranm/Documents/Python/First-Neural-Network/Tools"))
from HorizontalLineEnv import SantaFeTrailEnv
from ReplayBuffer import PrioritizedReplayBuffer
from ModelUtilities import save_stats, save_best_transitions, load_best_transitions, save_model, load_model


# Get the absolute path to the directory containing the script
script_dir = os.path.dirname(__file__)

# Get the path to the root directory (assuming your script is 2 levels deep from root)
# Adjust '..' based on how many levels up you need to go from your_script.py
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Construct the path to the Tools folder
tools_path = os.path.join(root_dir, 'Tools')

# Add the Tools folder to sys.path
sys.path.append(tools_path)
print(f"Added Tools directory to sys.path: {tools_path}")

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


# Import SNN model and utils (to be created)
from snn_model import SantaFeSNN
from config import num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, learning_rate, replay_buffer_size, recent_buffer_size, hidden_size, target_update_freq, n_step
from HorizontalLineEnv import original_trail

# Registering custom environment
gym.register(
    id="gymnasium_env/HorizontalLine-v0",
    entry_point="HorizontalLineEnv:SantaFeTrailEnv",
    reward_threshold=len(original_trail),
    max_episode_steps=int(len(original_trail)*1.5)
)

env = gym.make("gymnasium_env/HorizontalLine-v0", render_mode="rgb_array")

video_folder = "./videos"
env = RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=lambda episode_id: episode_id % 50 == 0,
    name_prefix="SantaFeSNN"
)

replay_buffer = deque(maxlen=replay_buffer_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

observation_shape = env.observation_space.shape
if len(observation_shape) != 1:
    raise ValueError(f"Expected observation_space.shape to be 1D (e.g., (input_dim,)), got {observation_shape}")
num_actions = env.action_space.n
model = SantaFeSNN(observation_shape, hidden_size, num_actions).to(device)
target_model = SantaFeSNN(observation_shape, hidden_size, num_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epsilon = epsilon_start

model_loaded, loaded_stats, epsilon = load_model(model, optimizer, epsilon_start)
if model_loaded:
    episode_stats = loaded_stats
    print("Loaded previous model and training stats")

best_transitions = load_best_transitions()
if best_transitions:
    for transition in best_transitions:
        replay_buffer.append(transition)
    print(f"Loaded {len(best_transitions)} best transitions")

best_episodes = []

step_count = 0
episode_stats = []
save_every = 50

for episode in trange(num_episodes, desc="Training"):

    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    done = False
    total_reward = 0
    last_turn_action = None  # Track last turn direction (1=left, 2=right)
    consecutive_turns = 0
    agent_positions = []  # Track agent's position at each timestep

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            obs_for_model = obs.unsqueeze(0) if obs.dim() == 2 else obs
            with torch.no_grad():
                q_values = model(obs_for_model)
                q_values = q_values.squeeze()
                if q_values.dim() != 1:
                    raise ValueError(f"After squeeze, unexpected q_values shape: {q_values.shape}")
                action = torch.argmax(q_values).item()

        # Record agent's position (must be in info['position'])
        if 'position' in info:
            agent_positions.append(info['position'])
        else:
            raise RuntimeError("info['position'] is missing. Environment must provide agent's true grid coordinates in info['position'].")

        # Track consecutive turns in the same direction (assuming 1=left, 2=right)
        if action in [1, 2]:
            if last_turn_action == action:
                consecutive_turns += 1
            else:
                consecutive_turns = 1
                last_turn_action = action
        else:
            consecutive_turns = 0
            last_turn_action = None

        # Discourage more than 10 consecutive turns in one direction
        if consecutive_turns > 8:
            reward = -200  # Penalty for spinning

        next_obs, reward, terminated, truncated, info = env.step(action) 
        done = done or terminated or truncated
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward += reward

        transition = (obs, action, reward, next_obs_tensor, done)
        replay_buffer.append(transition)
        obs = next_obs_tensor

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

            obs_batch = [torch.tensor(o, dtype=torch.float32, device=device) if not torch.is_tensor(o) else o for o in obs_batch]
            next_obs_batch = [torch.tensor(o, dtype=torch.float32, device=device) if not torch.is_tensor(o) else o for o in next_obs_batch]


            obs_batch = torch.cat(obs_batch)
            next_obs_batch = torch.cat(next_obs_batch)
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
            done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device)

            obs_batch = obs_batch.unsqueeze(1) if obs_batch.dim() == 2 else obs_batch
            next_obs_batch = next_obs_batch.unsqueeze(1) if next_obs_batch.dim() == 2 else next_obs_batch

            raw_model_out = model(obs_batch)
            if raw_model_out.dim() == 3:
                model_out = raw_model_out[:, -1, :]
            else:
                model_out = raw_model_out
            q_values = model_out.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                target_out = target_model(next_obs_batch)
                if target_out.dim() == 3:
                    target_out = target_out[:, -1, :]
                next_q_values = target_out.max(1)[0]
                targets = reward_batch + (1 - done_batch) * gamma * next_q_values

            loss = loss_fn(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        step_count += 1
        if step_count % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

    print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
    episode_stats.append({
        "episode": episode + 1,
        "total_reward": total_reward,
        "epsilon": epsilon
    })

    # --- Save agent positions for this episode ---
    positions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../Data/SantaFeTrail-SNN/Positions')
    os.makedirs(positions_dir, exist_ok=True)
    positions_path = os.path.join(positions_dir, f"episode_{episode+1}_positions.json")
    agent_positions_serializable = [pos.tolist() if hasattr(pos, "tolist") else pos for pos in agent_positions]
    with open(positions_path, "w") as f:
        json.dump(agent_positions_serializable, f)
    # --------------------------------------------

    # --- Collect transitions for best_episodes ---
    episode_length = len(agent_positions)
    episode_transitions = list(replay_buffer)[-episode_length:]
    best_episodes.append((total_reward, episode_transitions))
    # --------------------------------------------

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    if (episode + 1) % save_every == 0:
        save_model(model, optimizer, episode_stats, epsilon)
        stats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../Data/SantaFeTrail-SNN')
        os.makedirs(stats_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_path = os.path.join(stats_dir, f"episode_stats_{timestamp}.json")
        with open(stats_path, "w") as f:
            json.dump(episode_stats, f)
        print(f"Saved model and episode stats at episode {episode+1}")


if hasattr(env, "video_recorder") and env.video_recorder is not None:
    env.video_recorder.close()
    env.video_recorder = None

env.close()

best_episodes.sort(key=lambda x: x[0], reverse=True) # Sort by total reward
best_transitions = [] # Collect transitions from the best episodes
for _, transitions in best_episodes[:20]: # Take top 20 best episodes
    best_transitions.extend(transitions) # Ensure we don't exceed the buffer size

save_best_transitions(best_transitions) # Save best transitions to file
save_model(model, optimizer, episode_stats, epsilon) # Save final model
print(f"Saved {len(best_transitions)} transitions from best episodes") # Print number of transitions saved

stats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../Data/SantaFeTrail-SNN') # Ensure stats directory exists
os.makedirs(stats_dir, exist_ok=True) # Make directory if it doesn't exist
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Format timestamp for filename
stats_path = os.path.join(stats_dir, f"episode_stats_{timestamp}.json") # Save episode stats to file
with open(stats_path, "w") as f:
    json.dump(episode_stats, f)
print(f"Saved episode stats to {stats_path}") # Print path to saved stats file
