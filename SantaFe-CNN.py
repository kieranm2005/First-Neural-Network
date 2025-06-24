import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
'''To Do:
1. Vectorized Environment: Use `gym.vector.make` for parallel environments.
2. Collect statistics: Use `env.get_episode_stats()` to collect statistics.
3. Plotting: Use `matplotlib` to visualize the training progress.
4. Tune hyperparameters with Optuna'''

class SantaFeCNN(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(SantaFeCNN, self).__init__()
        # observation_shape is (channels, height, width), e.g., (6, 32, 32)
        channels, height, width = observation_shape

        # --- Convolutional Layers ---
        # Conv1: Input 6 channels, Output 32 channels. 3x3 kernel, stride 1, padding 1
        # Padding=1 maintains spatial dimensions (32x32 -> 32x32)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        # MaxPool1: Reduces spatial dimensions by factor of 2 (32x32 -> 16x16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2: Input 32 channels, Output 64 channels. 3x3 kernel, stride 1, padding 1
        # (16x16 -> 16x16)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # MaxPool2: Reduces spatial dimensions by factor of 2 (16x16 -> 8x8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv3: Input 64 channels, Output 128 channels. 3x3 kernel, stride 1, padding 1
        # (8x8 -> 8x8)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # MaxPool3: Reduces spatial dimensions by factor of 2 (8x8 -> 4x4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Calculate the size of the flattened features ---
        # After 3 pooling layers, each reducing dimension by 2, the original height/width are divided by 8.
        # Example: 32 / 2 / 2 / 2 = 4
        # So, the final feature map size will be 128 (out_channels of conv3) * 4 * 4
        flattened_size = 128 * (height // 8) * (width // 8) 

        # --- Fully Connected (Linear) Layers ---
        self.fc1 = nn.Linear(flattened_size, 512) # Example hidden layer size
        self.fc2 = nn.Linear(512, num_actions)    # Output layer for actions

    def forward(self, x):
        # Apply Conv -> ReLU -> Pool
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten the output for the fully connected layers
        # x.size(0) is the batch size
        x = x.view(x.size(0), -1) 
        
        # Apply Fully Connected -> ReLU
        x = F.relu(self.fc1(x))
        
        # Output layer 
        actions_logits = self.fc2(x)
        return actions_logits


# Registering custom environment
gym.register(
        id="gymnasium_env/SantaFeTrail-v0",
        entry_point="SantaFeTrailEnv:SantaFeTrailEnv",  # module:class
        reward_threshold=89,        
        max_episode_steps=600     
)

# Initializing environment
env = gym.make_vec("gymnasium_env/SantaFeTrail-v0", num_envs=5)

# Hyperparameters
num_episodes = 500
batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995
learning_rate = 1e-3
replay_buffer = deque(maxlen=50000)

# Model and optimizer
observation_shape = env.observation_space.shape  # (channels, height, width)
num_actions = env.action_space.n
model = SantaFeCNN(observation_shape, num_actions)
target_model = SantaFeCNN(observation_shape, num_actions)
target_model.load_state_dict(model.state_dict())
target_model.eval()
target_update_freq = 1000  # steps

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epsilon = epsilon_start

step_count = 0
for episode in range(num_episodes):
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # shape: (1, C, H, W)
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
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        total_reward += reward

        # Store transition in replay buffer
        replay_buffer.append((obs, action, reward, next_obs_tensor, done))

        obs = next_obs_tensor

        # Sample random minibatch and train
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
            obs_batch = torch.cat(obs_batch)
            action_batch = torch.tensor(action_batch)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
            next_obs_batch = torch.cat(next_obs_batch)
            done_batch = torch.tensor(done_batch, dtype=torch.float32)

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

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

env.close()