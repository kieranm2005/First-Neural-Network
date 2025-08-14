import gymnasium as gym
from FPGA.PolicyNetwork import PolicyNetwork
from FPGA.FPGAAgent import FPGAAgent
import torch.optim as optim
from Environments import HorizontalLineEnv

# To Do: STDP

env = HorizontalLineEnv()
agent = FPGAAgent("/dev/ttyUSB0")
policy = PolicyNetwork(obs_dim=1)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

episodes = 200
gamma = 0.99

for ep in range(episodes):
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)

    log_probs = []
    rewards = []

    done = False
    while not done:
        current_input = policy(obs)
        input_val = current_input.item()
        input_byte = int(input_val)

        spike = agent.step(input_byte)
        action = decode_spike(spike)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)

        # Compute log prob for REINFORCE
        log_prob = torch.log(current_input / 255 + 1e-6) if spike else torch.log(1 - current_input / 255 + 1e-6)
        log_probs.append(log_prob)
        rewards.append(reward)

        obs = next_obs
        done = terminated or truncated

    # REINFORCE update
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)

    loss = 0
    for log_prob, R in zip(log_probs, returns):
        loss -= log_prob * R

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {ep+1}/{episodes}, Total Reward: {sum(rewards)}")

agent.close()
env.close()
