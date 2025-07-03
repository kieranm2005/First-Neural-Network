import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque #double ended queue for efficient appends and pops
from torch.optim.lr_scheduler import StepLR
import json
import datetime

'''Overview:
1. Environment: Custom Santa Fe Trail environment using Gymnasium.
2. Model: Long Short-Term Memory (LSTM) neural network for sequential decision making.
3. Training: Reinforcement learning with prioritized experience replay.
4. Video Recording: Record episodes for visualization.
5. Hyperparameters: Tuned using Optuna for optimal performance.
6. Tensor shape compatibility: The model expects a 1D observation space, e.g., (input_dim,).
7. Observation: The observation is a single value indicating if there's food in front of the agent.'''

# Long Short-Term Memory Architecture for Santa Fe Trail
class SantaFeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SantaFeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size[0], hidden_size=hidden_size, batch_first=True)
        self.fc_value = nn.Linear(hidden_size, 128)
        self.fc_advantage = nn.Linear(hidden_size, 128)
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last time step
        value = F.relu(self.fc_value(x))
        advantage = F.relu(self.fc_advantage(x))
        value = self.value(value)
        advantage = self.advantage(advantage)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Registering custom environment
gym.register(
    id="gymnasium_env/SantaFeTrail-v0",
    entry_point="SantaFeTrailEnv:SantaFeTrailEnv",
    reward_threshold=89,
    max_episode_steps=150,
)

# Initializing environment
env = gym.make("gymnasium_env/SantaFeTrail-v0", render_mode="rgb_array")

video_folder = "./videos"
env = RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=lambda episode_id: episode_id % 200 == 0,
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
if len(observation_shape) != 1:
    raise ValueError(f"Expected observation_space.shape to be 1D (e.g., (input_dim,)), got {observation_shape}")
num_actions = env.action_space.n
hidden_size = 256
model = SantaFeLSTM(observation_shape, hidden_size, num_actions).to(device)
target_model = SantaFeLSTM(observation_shape, hidden_size, num_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

epsilon = epsilon_start
target_update_freq = 1000

# Prioritized Replay Buffer
# Number of transitions to keep in the buffer: 50000
# Transitions are tuples of (state, action, reward, next_state, done)
# Alpha controls the prioritization of experiences (0 < alpha <= 1)
# Beta controls the importance sampling correction (0 < beta <= 1)
# Capacity is the maximum number of transitions to store
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

def load_model(model, optimizer, filename="santa_fe_lstm.pt"):
    # Load the model and optimizer state
    try:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode_stats = checkpoint['episode_stats']
        epsilon = checkpoint['epsilon']
        print(f"Loaded model from {filename} with {len(episode_stats)} episodes")
        return True, episode_stats, epsilon
    except FileNotFoundError:
        print(f"No previous model found at {filename}, starting fresh")
        return False, [], epsilon_start

# Load previous model if it exists
model_loaded, loaded_stats, epsilon = load_model(model, optimizer)
if model_loaded:
    episode_stats = loaded_stats
    print("Loaded previous model and training stats")

def load_best_transitions(filename="best_transitions.pt"):
    """Load the best transitions from a file with error handling"""
    try:
        return torch.load(filename, weights_only=False)
    except Exception as e:
        print(f"Error loading transitions: {e}")
        return []

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
reward_threshold = 20  # Initial value

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
            obs_batch = torch.stack([torch.tensor(o, dtype=torch.float32, device=device).squeeze(0) 
                         if isinstance(o, np.ndarray) else o.squeeze(0) 
                         for o in obs_batch])
            next_obs_batch = torch.stack([torch.tensor(o, dtype=torch.float32, device=device).squeeze(0) 
                              if isinstance(o, np.ndarray) else o.squeeze(0) 
                              for o in next_obs_batch])
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
            done_batch = torch.tensor([float(d) for d in done_batch], dtype=torch.float32, device=device)

            # Q(s, a)
            q_values = model(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

            # max_a' Q(s', a')
            with torch.no_grad():
                next_q_values = target_model(next_obs_batch)
                max_next_q_values = next_q_values.max(1)[0]
                target = reward_batch + (1 - done_batch) * gamma * max_next_q_values

            # Compute TD errors and loss
            td_errors = q_values - target
            per_sample_loss = td_errors.pow(2)
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            loss = (per_sample_loss * weights_tensor).mean()

            # Update network
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update priorities
            new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
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

def save_stats(stats):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_stats_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(stats, f)

save_stats(episode_stats)

def save_best_transitions(transitions, filename="best_transitions.pt"):
    """Save the best transitions to a file with error handling"""
    try:
        # Convert tensors to CPU before saving
        processed_transitions = []
        for state, action, reward, next_state, done in transitions:
            processed_transitions.append((
                state.cpu().numpy() if torch.is_tensor(state) else state,
                action,
                reward,
                next_state.cpu().numpy() if torch.is_tensor(next_state) else next_state,
                done
            ))
        torch.save(processed_transitions, filename)
        print(f"Successfully saved {len(processed_transitions)} transitions to {filename}")
    except Exception as e:
        print(f"Error saving transitions: {e}")

def load_best_transitions(filename="best_transitions.pt"):
    """Load the best transitions from a file with error handling"""
    try:
        return torch.load(filename, weights_only=False)
    except Exception as e:
        print(f"Error loading transitions: {e}")
        return []

def save_model(model, optimizer, episode_stats, filename="santa_fe_lstm.pt"):
    # Save the model, optimizer state, and training stats
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_stats': episode_stats,
        'epsilon': epsilon
    }, filename)

def load_model(model, optimizer, filename="santa_fe_lstm.pt"):
    # Load the model and optimizer state
    try:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode_stats = checkpoint['episode_stats']
        epsilon = checkpoint['epsilon']
        return True, episode_stats, epsilon
    except FileNotFoundError:
        return False, [], epsilon_start

# Add after the episode loop (before env.close())

# Sort episodes by reward and save the transitions from the best 200 episodes
best_episodes.sort(key=lambda x: x[0], reverse=True)
best_transitions = []
for _, transitions in best_episodes[:200]:
    best_transitions.extend(transitions)

# Save best transitions and final model
save_best_transitions(best_transitions)
save_model(model, optimizer, episode_stats)
print(f"Saved {len(best_transitions)} transitions from best episodes")
