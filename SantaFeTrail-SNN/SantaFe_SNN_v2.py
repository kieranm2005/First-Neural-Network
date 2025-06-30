import snntorch as snn
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from SantaFeTrailEnv import SantaFeTrailEnv

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 16  # 4x4 grid
HIDDEN_SIZE = 64
OUTPUT_SIZE = 4  # up, down, left, right
NUM_STEPS = 25
GAMMA = 0.99
BETA = 0.5  # For RLeaky neurons

# Spiking Neural Network definition
class SNNBase(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.hidden1 = snn.RLeaky(beta=BETA, linear_features=hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = snn.RLeaky(beta=BETA, linear_features=hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, num_steps):
        batch_size = x.shape[0]
        mem1 = torch.zeros((batch_size, self.fc1.out_features), device=x.device)
        mem2 = torch.zeros((batch_size, self.fc2.out_features), device=x.device)
        spk_sum2 = torch.zeros_like(mem2)

        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.hidden1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.hidden2(cur2, mem2)
            spk_sum2 += spk2

        return spk_sum2 / num_steps

class SNN(SNNBase):
    def __init__(self, input_size, hidden_size, output_size, num_steps=NUM_STEPS):
        super().__init__(input_size, hidden_size, output_size)
        self.num_steps = num_steps

    def forward(self, x):
        spk_avg = super().forward(x, self.num_steps)
        return self.fc3(spk_avg)

class PolicySNN(SNNBase):
    def __init__(self, input_size, hidden_size, output_size, num_steps=NUM_STEPS):
        super().__init__(input_size, hidden_size, output_size)
        self.num_steps = num_steps

    def forward(self, x):
        spk_avg = super().forward(x, self.num_steps)
        return torch.softmax(self.fc3(spk_avg), dim=-1)

class PrioritizedReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = deque()
        self.min_priority = 1e-6

    def add(self, experience, priority):
        priority = max(priority, self.min_priority)
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.popleft()
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        priorities = np.array(self.priorities, dtype=np.float64)
        total = priorities.sum()

        if total <= 0 or np.isnan(total) or np.isinf(total):
            probs = np.ones(len(self.buffer)) / len(self.buffer)
        else:
            probs = priorities / total

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = max(priority, self.min_priority)

class SNNTrainer:
    def __init__(self, env, algorithm='dqn'):
        self.env = env
        self.algorithm = algorithm
        self.obs_shape = env.observation_space.shape
        self.input_size = np.prod(self.obs_shape)
        self.output_size = env.action_space.n

        if algorithm == 'dqn':
            self.model = SNN(self.input_size, HIDDEN_SIZE, self.output_size).to(DEVICE)
            self.target_model = SNN(self.input_size, HIDDEN_SIZE, self.output_size).to(DEVICE)
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            self.model = PolicySNN(self.input_size, HIDDEN_SIZE, self.output_size).to(DEVICE)

        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.replay_buffer = PrioritizedReplay(capacity=50000)

        # Running statistics for normalization
        self.running_mean = torch.zeros(self.input_size, device=DEVICE)
        self.running_var = torch.ones(self.input_size, device=DEVICE)
        self.momentum = 0.99

    def update_running_stats(self, x):
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        self.running_mean.mul_(self.momentum).add_(batch_mean * (1 - self.momentum))
        self.running_var.mul_(self.momentum).add_(batch_var * (1 - self.momentum))
        return (x - self.running_mean) / torch.sqrt(self.running_var + 1e-8)

    def get_action(self, state, epsilon):
        if self.algorithm == 'reinforce':
            with torch.no_grad():
                probs = self.model(state)
                dist = torch.distributions.Categorical(probs)
                return dist.sample().item()
        else:
            if random.random() > epsilon:
                with torch.no_grad():
                    q_values = self.model(state)
                    if random.random() < 0.3:  # occasional noise
                        q_values += torch.randn_like(q_values) * 0.1
                    return q_values.argmax().item()
            return self.env.action_space.sample()

    def train_dqn(self, num_episodes=1000, batch_size=128, learning_rate=0.0003,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.997):
        print("Training SNN with DQN...")
        epsilon = epsilon_start
        target_update_freq = 1000
        step_counter = 0
        best_reward = float('-inf')
        episode_rewards = []
        episode_epsilons = []
        patience = 50
        no_improve = 0
        best_reward_early_stopping = float('-inf')

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                obs_flat = np.array(obs).flatten()
                obs_tensor = self.update_running_stats(
                    torch.tensor(obs_flat, dtype=torch.float32, device=DEVICE)
                ).unsqueeze(0)

                action = self.get_action(obs_tensor, epsilon)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                reward = np.clip(reward, -1, 1)

                next_obs_flat = np.array(next_obs).flatten()
                with torch.no_grad():
                    obs_tensor = torch.tensor([obs_flat], dtype=torch.float32, device=DEVICE)
                    next_obs_tensor = torch.tensor([next_obs_flat], dtype=torch.float32, device=DEVICE)
                    q_value = self.model(obs_tensor).gather(1, torch.tensor([[action]], device=DEVICE)).squeeze()
                    next_q_values = self.target_model(next_obs_tensor).max(1)[0]
                    target = torch.tensor(reward, dtype=torch.float32, device=DEVICE) + GAMMA * next_q_values * (1 - torch.tensor(done, dtype=torch.float32, device=DEVICE))
                    td_error = torch.abs(q_value - target).item()

                self.replay_buffer.add((obs_flat, action, reward, next_obs_flat, done), td_error)
                obs = next_obs

                if len(self.replay_buffer.buffer) >= batch_size:
                    self._update_dqn(batch_size)

                if step_counter % target_update_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                step_counter += 1

            episode_rewards.append(total_reward)
            episode_epsilons.append(epsilon)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            self.scheduler.step()

            # Early stopping and best model saving 
            avg_reward = np.mean(episode_rewards[-100:])
            if avg_reward > best_reward_early_stopping:
                best_reward_early_stopping = avg_reward
                no_improve = 0
                torch.save(self.model.state_dict(), 'best_snn_model.pt')
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered")
                    break

            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}, Best Reward: {max(episode_rewards)}")

        return episode_rewards, episode_epsilons

    def _update_dqn(self, batch_size):
        batch, indices = self.replay_buffer.sample(batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=DEVICE)
        action_batch = torch.tensor(action_batch, device=DEVICE)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=DEVICE)
        next_obs_batch = torch.tensor(np.stack(next_obs_batch), dtype=torch.float32, device=DEVICE)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=DEVICE)

        q_values = self.model(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.model(next_obs_batch).argmax(1)
            next_q_values = self.target_model(next_obs_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = reward_batch + GAMMA * next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        with torch.no_grad():
            td_errors = torch.abs(q_values - target).cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

    def train_reinforce(self, num_episodes=1000, learning_rate=0.001, entropy_weight=0.01):
        print("Training SNN with REINFORCE...")
        episode_rewards = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            total_reward = 0

            while not done:
                obs_flat = np.array(obs).flatten()
                states.append(obs_flat)

                obs_tensor = torch.FloatTensor(obs_flat).to(DEVICE).unsqueeze(0)
                with torch.no_grad():
                    probs = self.model(obs_tensor)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                actions.append(action)
                rewards.append(reward)
                total_reward += reward
                obs = next_obs

            # Compute returns and update policy
            self._update_reinforce(states, actions, rewards, entropy_weight)
            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        return episode_rewards, []

    def _update_reinforce(self, states, actions, rewards, entropy_weight):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        if len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        self.optimizer.zero_grad()
        total_loss = 0
        entropy_loss = 0

        for state, action, ret in zip(states, actions, returns_tensor):
            state_tensor = torch.FloatTensor(state).to(DEVICE).unsqueeze(0)
            probs = self.model(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action_tensor = torch.tensor(action, device=DEVICE)
            log_prob = dist.log_prob(action_tensor)
            loss = -log_prob * ret.item()
            entropy = dist.entropy()
            entropy_loss += -entropy
            total_loss += loss + entropy_weight * entropy_loss

        total_loss.backward()
        self.optimizer.step()

def plot_results(episode_rewards, episode_epsilons, algorithm):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title(f"{algorithm} Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(episode_epsilons)
    plt.title(f"{algorithm} Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig(f"{algorithm}_training_results.png")
    plt.show()

def main():
    # Register and create environment
    gym.register(
        id="gymnasium_env/SantaFeTrail-v0",
        entry_point="SantaFeTrailEnv:SantaFeTrailEnv",
        reward_threshold=89,
        max_episode_steps=600
    )
    env = gym.make("gymnasium_env/SantaFeTrail-v0")

    # Train with REINFORCE
    trainer = SNNTrainer(env, algorithm='reinforce')
    episode_rewards, episode_epsilons = trainer.train_reinforce()
    plot_results(episode_rewards, episode_epsilons, 'REINFORCE')

    # Save results
    np.save("episode_rewards_reinforce.npy", episode_rewards)
    np.save("episode_epsilons_reinforce.npy", episode_epsilons)

    env.close()

if __name__ == "__main__":
    main()
