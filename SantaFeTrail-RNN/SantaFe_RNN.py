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

# LSTM Architecture for Santa Fe Trail
class SantaFeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SantaFeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size[0], hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.layer_norm = nn.LayerNorm(128)  # Changed from BatchNorm1d

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last time step
        x = F.relu(self.fc1(x))
        x = self.layer_norm(x)  # Changed from self.batch_norm(x)
        x = self.fc2(x)
        return x

# Registering custom environment
gym.register(
    id="gymnasium_env/SantaFeTrail-v0",
    entry_point="SantaFeTrailEnv:SantaFeTrailEnv",
    reward_threshold=89,
    max_episode_steps=150,
)

# Initializing environment
env = gym.make("gymnasium_env/SantaFeTrail-v0", render_mode="rgb_array")

def video_trigger(episode_id):
    return episode_id % 200 == 0

video_folder = "./videos"
env = RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=video_trigger,
    name_prefix="SantaFeLSTM"
)

# Hyperparameters
num_episodes = 600
batch_size = 64
gamma = 0.9006346496123904
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.997
learning_rate = 0.0011779172637528985
replay_buffer = deque(maxlen=50000)
recent_buffer = deque(maxlen=5000)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and optimizer
observation_shape = env.observation_space.shape
num_actions = env.action_space.n
hidden_size = 128
model = SantaFeLSTM(observation_shape, hidden_size, num_actions).to(device)
target_model = SantaFeLSTM(observation_shape, hidden_size, num_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

epsilon = epsilon_start
target_update_freq = 1000

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < self.capacity:
            probs = self.priorities[:len(self.buffer)]
        else:
            probs = self.priorities

        probs = probs ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** -self.beta
        weights = weights / weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

# Initialize Prioritized Replay Buffer
prioritized_replay_buffer = PrioritizedReplayBuffer(capacity=50000)

# Training loop
step_count = 0
episode_stats = []
reward_threshold = 20  # Initial value

n_step = 4
n_step_buffer = deque(maxlen=n_step)
for episode in range(num_episodes):
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    done = False
    total_reward = 0
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
        replay_buffer.append((obs, action, reward, next_obs_tensor, done))
        recent_buffer.append((obs, action, reward, next_obs_tensor, done))
        prioritized_replay_buffer.add(obs, action, reward, next_obs_tensor, done)
        obs = next_obs_tensor

        # N-step bootstrapping
        n_step_buffer.append((obs, action, reward, next_obs_tensor, done))
        if len(n_step_buffer) == n_step:
            R = sum([n_step_buffer[i][2] * (gamma ** i) for i in range(n_step) if not n_step_buffer[i][4]])
            state, action, _, _, _ = n_step_buffer[0]
            _, _, _, next_state, done = n_step_buffer[-1]
            replay_buffer.append((state, action, R, next_state, done))
            prioritized_replay_buffer.add(state, action, R, next_state, done)

        # Sample random minibatch and train
        if len(replay_buffer) >= batch_size:
            batch, indices, weights = prioritized_replay_buffer.sample(batch_size)
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

            obs_batch = torch.cat(obs_batch, dim=0)
            next_obs_batch = torch.cat(next_obs_batch, dim=0)
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
            done_batch = torch.tensor([float(d) for d in done_batch], dtype=torch.float32, device=device)

            # Q(s, a)
            q_values = model(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

            # max_a' Q(s', a')
            with torch.no_grad():
                next_actions = model(next_obs_batch).argmax(1)
                next_q_values = target_model(next_obs_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target = reward_batch + gamma * next_q_values * (1 - done_batch)

            loss = loss_fn(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

env.close()

def save_stats(stats):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_stats_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(stats, f)

save_stats(episode_stats)
