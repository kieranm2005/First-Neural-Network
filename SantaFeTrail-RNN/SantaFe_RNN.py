import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class SantaFeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SantaFeRNN, self).__init__()
        channels, height, width = input_size
        # --- CNN feature extractor (copied from SantaFeCNN) ---
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Calculate flattened CNN feature size
        cnn_flattened_size = 128 * (height // 8) * (width // 8)
        # --- RNN ---
        self.rnn = nn.RNN(input_size=cnn_flattened_size, hidden_size=hidden_size, batch_first=True)
        # --- FC layers ---
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.size()
        # Reshape to (batch_size * seq_len, channels, height, width)
        x = x.view(batch_size * seq_len, channels, height, width)
        # Pass each frame through CNN
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # (batch_size * seq_len, cnn_flattened_size)
        # Reshape to (batch_size, seq_len, cnn_flattened_size)
        x = x.view(batch_size, seq_len, -1)
        # Pass through RNN
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Take last time step
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Registering custom environment
gym.register(
        id="gymnasium_env/SantaFeTrail-v0",
        entry_point="SantaFeTrailEnv:SantaFeTrailEnv",  
        reward_threshold=89,        
        max_episode_steps=150     
)

# Initializing environment
env = gym.make("gymnasium_env/SantaFeTrail-v0")

# Hyperparameters
num_episodes = 10000
batch_size = 64
gamma = 0.9006346496123904
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.9910256339861461
learning_rate = 0.0011779172637528985
replay_buffer = deque(maxlen=50000)
recent_buffer = deque(maxlen=5000)  # smaller buffer for recent transitions

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and optimizer
observation_shape = env.observation_space.shape  # (channels, height, width)
num_actions = env.action_space.n
hidden_size = 256 
model = SantaFeRNN(observation_shape, hidden_size, num_actions).to(device)
target_model = SantaFeRNN(observation_shape, hidden_size, num_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epsilon = epsilon_start
target_update_freq = 1000  # steps

# Training loop
step_count = 0
episode_stats = []
for episode in range(num_episodes):
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
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
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
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
            obs_batch = torch.cat([o.to(device) for o in obs_batch])  # (batch, 1, C, H, W)
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
            next_obs_batch = torch.cat([no.to(device) for no in next_obs_batch])
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