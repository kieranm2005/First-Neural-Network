import snntorch as snn
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random
from SantaFeTrailEnv import SantaFeTrailEnv  

# Summary of architecture:
# - Input layer: 16 input neurons (4x4 grid)
# - Hidden layer 1: 32 RLeaky spiking neurons
# - Hidden layer 2: 32 RLeaky spiking neurons
# - Output layer: 4 output neurons (up, down, left, right)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Spiking Neural Network definition using RLeaky neurons
class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_steps=25):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden1 = snn.RLeaky(beta=0.5, linear_features=hidden_size)  # Specify linear_features
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = snn.RLeaky(beta=0.5, linear_features=hidden_size)  # Specify linear_features
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.num_steps = num_steps  # Number of time steps for SNN simulation

    def forward(self, x):
        batch_size = x.shape[0]
        # Initialize membrane potentials and spike accumulators
        mem1 = torch.zeros((batch_size, self.fc1.out_features), device=x.device)
        mem2 = torch.zeros((batch_size, self.fc2.out_features), device=x.device)
        spk_sum2 = torch.zeros_like(mem2)
        
        # Simulate SNN over multiple time steps
        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.hidden1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.hidden2(cur2, mem2)
            spk_sum2 += spk2  # Accumulate spikes
        
        # Average spikes and pass through final layer
        out = self.fc3(spk_sum2 / self.num_steps)
        return out

# Policy Network for REINFORCE (outputs action probabilities)
class PolicySNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_steps=25):
        super(PolicySNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden1 = snn.RLeaky(beta=0.5, linear_features=hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = snn.RLeaky(beta=0.5, linear_features=hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.num_steps = num_steps

    def forward(self, x):
        batch_size = x.shape[0]
        mem1 = torch.zeros((batch_size, self.fc1.out_features), device=x.device)
        mem2 = torch.zeros((batch_size, self.fc2.out_features), device=x.device)
        spk_sum2 = torch.zeros_like(mem2)
        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.hidden1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.hidden2(cur2, mem2)
            spk_sum2 += spk2
        out = self.fc3(spk_sum2 / self.num_steps)
        return torch.softmax(out, dim=-1)  # Output action probabilities

# Prioritized Experience Replay Buffer
class PrioritizedReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        
    def add(self, experience, priority):
        min_priority = 1e-6
        priority = max(priority, min_priority)  # Avoid zero priorities
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size):
        priorities = np.array(self.priorities, dtype=np.float64)
        total = priorities.sum()
        if total == 0 or np.isnan(total) or np.isinf(total):
            # fallback to uniform sampling if priorities are invalid
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        else:
            probs = priorities / total
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices

# Main training loop for SNN agent

def train_snn(env, num_episodes=1000, batch_size=128, learning_rate=0.0003, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.997, num_steps=50):
    print("Training SNN (RLeaky) on Santa Fe Trail environment...")
    obs_shape = env.observation_space.shape
    input_size = np.prod(obs_shape)
    hidden_size = 64         # Larger hidden layer
    output_size = env.action_space.n

    # Initialize main and target SNN models
    model = SNN(input_size, hidden_size, output_size, num_steps=num_steps).to(device)
    target_model = SNN(input_size, hidden_size, output_size, num_steps=num_steps).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = deque(maxlen=50000)
    epsilon = epsilon_start
    target_update_freq = 1000  # Update target network every 1000 steps
    step_counter = 0
    prioritized_replay = PrioritizedReplay(capacity=50000) # Prioritized Replay buffer
    # Learning rate scheduler for gradual decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # Track metrics for monitoring
    running_reward = deque(maxlen=100)
    best_reward = float('-inf')

    episode_rewards = []
    episode_epsilons = []

    patience = 50  # Early stopping patience
    no_improve = 0
    best_reward_early_stopping = float('-inf')

    # Running state normalization statistics (persist across episodes, on device)
    state_mean = torch.zeros(input_size, device=device)
    state_std = torch.ones(input_size, device=device)
    
    def normalize_state(state):
        state_tensor = torch.FloatTensor(state).to(device)
        return (state_tensor - state_mean) / (state_std + 1e-8)
    
    # Epsilon-greedy action selection with some noise
    def get_action(state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = model(state)
                if random.random() < 0.3:  # Add occasional noise even during exploitation
                    noise = torch.randn_like(q_values) * 0.1
                    q_values += noise
                return q_values.argmax().item()
        # Random exploration
        return env.action_space.sample()

    # Training loop over episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            if isinstance(obs, tuple):
                obs = obs[0]
            obs_flat = np.array(obs).flatten()

            # Update running mean and std for normalization (exponential moving average, in-place)
            obs_flat_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=device)
            state_mean.copy_(0.99 * state_mean + 0.01 * obs_flat_tensor)
            state_std.copy_(0.99 * state_std + 0.01 * (obs_flat_tensor - state_mean) ** 2)
            obs_tensor = normalize_state(obs_flat).to(device).unsqueeze(0)
            action = get_action(obs_tensor, epsilon)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            next_obs_flat = np.array(next_obs).flatten()

            # Calculate TD error for prioritization
            with torch.no_grad():
                obs_tensor = torch.tensor([obs_flat], dtype=torch.float32, device=device)
                next_obs_tensor = torch.tensor([next_obs_flat], dtype=torch.float32, device=device)
                q_value = model(obs_tensor).gather(1, torch.tensor([[action]], device=device)).squeeze()
                next_q_values = target_model(next_obs_tensor).max(1)[0]
                target = torch.tensor(reward, dtype=torch.float32, device=device) + gamma * next_q_values * (1 - torch.tensor(done, dtype=torch.float32, device=device))
                td_error = torch.abs(q_value - target).item()
            prioritized_replay.add((obs_flat, action, reward, next_obs_flat, done), priority=td_error)

            obs = next_obs  # Move to next state

            # Learning step if enough samples in buffer
            if len(prioritized_replay.buffer) >= batch_size:
                batch, indices = prioritized_replay.sample(batch_size)
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
                obs_batch = np.stack(obs_batch)  # Ensures a proper 2D array
                obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                action_batch = torch.tensor(action_batch, device=device)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
                next_obs_batch = np.stack(next_obs_batch)
                next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32, device=device)
                done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device)

                # Compute Q-values and targets
                q_values = model(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_actions = model(next_obs_batch).argmax(1)
                    next_q_values = target_model(next_obs_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target = reward_batch + gamma * next_q_values * (1 - done_batch)

                # Compute loss and update model
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update priorities in the prioritized replay buffer
                with torch.no_grad():
                    td_errors = torch.abs(q_values - target).cpu().numpy()
                for idx, error in zip(indices, td_errors):
                    prioritized_replay.priorities[idx] = error
            # Periodically update target network
            if step_counter % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
            step_counter += 1

            # Periodically update target network (skip first step)
            if step_counter != 0 and step_counter % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
            step_counter += 1
        episode_rewards.append(total_reward)
        episode_epsilons.append(epsilon)
        running_reward.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}, Best Reward: {best_reward}")
        # Early stopping based on running average reward
        avg_reward = np.mean(episode_rewards[-100:])
        if avg_reward > best_reward_early_stopping:
            best_reward_early_stopping = avg_reward
            no_improve = 0
            torch.save(model.state_dict(), 'best_snn_model.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered")
                break

    print("Training complete.")
    env.close()
    return episode_rewards, episode_epsilons

# REINFORCE training loop
def train_reinforce(env, num_episodes=1000, learning_rate=0.0003, gamma=0.99, num_steps=25):
    print("Training SNN (RLeaky) with REINFORCE on Santa Fe Trail environment...")
    obs_shape = env.observation_space.shape
    input_size = np.prod(obs_shape)
    hidden_size = 64
    output_size = env.action_space.n

    policy = PolicySNN(input_size, hidden_size, output_size, num_steps=num_steps).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    episode_rewards = []
    episode_epsilons = []  # Not used in REINFORCE, but kept for plotting

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        log_probs = []
        rewards = []
        total_reward = 0

        while not done:
            obs_flat = np.array(obs).flatten()
            obs_tensor = torch.FloatTensor(obs_flat).to(device).unsqueeze(0)
            probs = policy(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            obs = next_obs

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize

        # Policy loss
        loss = -torch.stack(log_probs) * returns
        loss = loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(total_reward)
        episode_epsilons.append(0)  # No epsilon in REINFORCE

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    print("Training complete.")
    env.close()
    return episode_rewards, episode_epsilons

if __name__ == "__main__":
    # Register custom Santa Fe Trail environment with Gymnasium
    gym.register(
        id="gymnasium_env/SantaFeTrail-v0",
        entry_point="SantaFeTrailEnv:SantaFeTrailEnv",  # module:class
        reward_threshold=89,        
        max_episode_steps=600      # Increased from 100 to allow more moves
    )

    # Initialize environment
    env = gym.make("gymnasium_env/SantaFeTrail-v0")

    # Before training, collect N random states and compute mean/std
    random_states = []
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        random_states.append(np.array(obs).flatten())
        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()
    random_states = np.stack(random_states)
    state_mean = torch.tensor(random_states.mean(axis=0), device=device)
    state_std = torch.tensor(random_states.std(axis=0) + 1e-8, device=device)

    # Train the SNN agent
    episode_rewards, episode_epsilons = train_snn(env)

    # Save results to file
    np.save("episode_rewards.npy", episode_rewards)
    np.save("episode_epsilons.npy", episode_epsilons)

    # Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.subplot(1, 2, 2)
    plt.plot(episode_epsilons)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()

    # Train the SNN agent with REINFORCE
    episode_rewards_reinforce, episode_epsilons_reinforce = train_reinforce(env)

    # Save REINFORCE results to file
    np.save("episode_rewards_reinforce.npy", episode_rewards_reinforce)
    np.save("episode_epsilons_reinforce.npy", episode_epsilons_reinforce)

    # Plot REINFORCE results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards_reinforce)
    plt.title("REINFORCE Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.subplot(1, 2, 2)
    plt.plot(episode_epsilons_reinforce)
    plt.title("REINFORCE Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.tight_layout()
    plt.savefig("reinforce_training_results.png")
    plt.show()
