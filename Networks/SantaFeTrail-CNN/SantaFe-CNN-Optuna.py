import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import optuna

from SantaFe_CNN_v2 import SantaFeCNN  # Adjust this import if needed

def objective(trial):
    # Hyperparameter search space
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    gamma = trial.suggest_uniform('gamma', 0.90, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.990, 0.9999)

    # Environment and model setup
    env = gym.make("gymnasium_env/SantaFeTrail-v0")
    observation_shape = env.observation_space.shape
    num_actions = env.action_space.n
    model = SantaFeCNN(observation_shape, num_actions)
    target_model = SantaFeCNN(observation_shape, num_actions)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    replay_buffer = deque(maxlen=50000)
    recent_buffer = deque(maxlen=5000)
    epsilon = 1.0
    epsilon_end = 0.1
    num_episodes = 30  # Fewer episodes for faster tuning
    target_update_freq = 1000
    total_rewards = []

    step_count = 0
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(obs)
                    action = torch.argmax(q_values, dim=1).item()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            total_reward += reward
            replay_buffer.append((obs, action, reward, next_obs_tensor, done))
            recent_buffer.append((obs, action, reward, next_obs_tensor, done))
            obs = next_obs_tensor

            if len(replay_buffer) >= batch_size:
                if len(recent_buffer) >= batch_size // 2:
                    batch = random.sample(replay_buffer, batch_size // 2) + random.sample(recent_buffer, batch_size // 2)
                else:
                    batch = random.sample(replay_buffer, batch_size)
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
                obs_batch = torch.cat(obs_batch)
                action_batch = torch.tensor(action_batch)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
                next_obs_batch = torch.cat(next_obs_batch)
                done_batch = torch.tensor(done_batch, dtype=torch.float32)
                q_values = model(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
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

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        total_rewards.append(total_reward)

    env.close()
    # Return average reward over last 5 episodes
    return np.mean(total_rewards[-5:])

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best hyperparameters:", study.best_params)