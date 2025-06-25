import snntorch as snn
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random
from SantaFeTrailEnv import SantaFeTrailEnv  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_steps=25):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden = snn.Leaky(beta=0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.num_steps = num_steps

    def forward(self, x):
        # x: [batch, input_size]
        batch_size = x.shape[0]
        mem = torch.zeros((batch_size, self.fc1.out_features), device=x.device)
        spk_sum = torch.zeros((batch_size, self.fc1.out_features), device=x.device)
        for _ in range(self.num_steps):
            cur = self.fc1(x)
            spk, mem = self.hidden(cur, mem)
            spk_sum += spk
        out = self.fc2(spk_sum / self.num_steps)  # mean spike count
        return out

def train_snn(env, num_episodes=500, batch_size=64, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, num_steps=25):
    print("Training SNN on Santa Fe Trail environment...")
    obs_shape = env.observation_space.shape
    input_size = np.prod(obs_shape)
    hidden_size = 128
    output_size = env.action_space.n

    model = SNN(input_size, hidden_size, output_size, num_steps=num_steps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = deque(maxlen=50000)
    epsilon = epsilon_start

    episode_rewards = []
    episode_epsilons = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            if isinstance(obs, tuple):
                obs = obs[0]
            obs_flat = np.array(obs).flatten()
            obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action_logits = model(obs_tensor)
                    action = torch.argmax(action_logits, dim=1).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            next_obs_flat = np.array(next_obs).flatten()
            replay_buffer.append((obs_flat, action, reward, next_obs_flat, done))
            obs = next_obs

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
                obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float32, device=device)
                action_batch = torch.tensor(action_batch, device=device)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
                next_obs_batch = torch.tensor(np.array(next_obs_batch), dtype=torch.float32, device=device)
                done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device)

                q_values = model(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = model(next_obs_batch).max(1)[0]
                target = reward_batch + gamma * next_q_values * (1 - done_batch)

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        episode_epsilons.append(epsilon)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
    print("Training complete.")
    env.close()
    return episode_rewards, episode_epsilons

if __name__ == "__main__":
    # Registering custom environment
    gym.register(
        id="gymnasium_env/SantaFeTrail-v0",
        entry_point="SantaFeTrailEnv:SantaFeTrailEnv",  # module:class
        reward_threshold=89,        
        max_episode_steps=100     
)

    # Initializing environment
    env = gym.make("gymnasium_env/SantaFeTrail-v0")


    # Training the SNN
    train_snn(env)