import sys
import os
sys.path.append(os.path.abspath("/u/kieranm/Documents/Python/First-Neural-Network/Environments"))  # Add absolute Environments path to sys.path

import optuna
import torch
import numpy as np
import random
import datetime
import json
from collections import deque
from tqdm import trange

from rnn_model import SantaFeLSTM
from replay_buffer import PrioritizedReplayBuffer
from utils import save_stats, save_best_transitions, load_best_transitions, save_model, load_model
from config import num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, learning_rate, replay_buffer_size, recent_buffer_size, hidden_size, target_update_freq, n_step
from HorizontalLineEnv import original_trail, SantaFeTrailEnv
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics



# Register custom environment
if not hasattr(gym.envs.registration.registry, 'env_specs') or 'gymnasium_env/HorizontalLine-v0' not in gym.envs.registration.registry.env_specs:
    gym.register(
        id="gymnasium_env/HorizontalLine-v0",
        entry_point="HorizontalLineEnv:SantaFeTrailEnv",
        reward_threshold=len(original_trail),
        max_episode_steps=int(len(original_trail)*1.7)
    )

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    gamma = trial.suggest_uniform('gamma', 0.90, 0.99)
    epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.90, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    target_update_freq = trial.suggest_categorical('target_update_freq', [100, 500, 1000])

    env = gym.make("gymnasium_env/HorizontalLine-v0")
    replay_buffer = deque(maxlen=replay_buffer_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observation_shape = env.observation_space.shape
    num_actions = env.action_space.n
    model = SantaFeLSTM(observation_shape, hidden_size, num_actions).to(device)
    target_model = SantaFeLSTM(observation_shape, hidden_size, num_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epsilon = epsilon_start
    step_count = 0
    episode_stats = []
    best_reward = -float('inf')

    for episode in range(20):  # Use fewer episodes for faster Optuna trials
        obs, info = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                obs_for_model = obs.unsqueeze(0) if obs.dim() == 2 else obs
                with torch.no_grad():
                    q_values = model(obs_for_model)
                    action = torch.argmax(q_values, dim=1).item()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            total_reward += reward
            transition = (obs, action, reward, next_obs_tensor, done)
            replay_buffer.append(transition)
            obs = next_obs_tensor
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
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
        episode_stats.append({"episode": episode + 1, "total_reward": total_reward, "epsilon": epsilon})
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if total_reward > best_reward:
            best_reward = total_reward
    env.close()
    return best_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Reward: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    # Save best params
    with open("best_optuna_params.json", "w") as f:
        json.dump(trial.params, f, indent=2)
