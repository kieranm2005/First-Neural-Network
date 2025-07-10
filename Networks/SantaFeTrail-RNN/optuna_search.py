import optuna
import torch
import numpy as np
from rnn_model import SantaFeLSTM
from replay_buffer import PrioritizedReplayBuffer
from config import batch_size, replay_buffer_size
import gymnasium as gym

# Register custom environment
gym.register(
    id="gymnasium_env/SantaFeTrail-v0",
    entry_point="SantaFeTrailEnv:SantaFeTrailEnv",
    reward_threshold=89,
    max_episode_steps=150,
)

def objective(trial):
    gamma = trial.suggest_float("gamma", 0.85, 0.99)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.95, 0.999)
    n_step = trial.suggest_int("n_step", 1, 8)

    env = gym.make("gymnasium_env/SantaFeTrail-v0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observation_shape = env.observation_space.shape
    num_actions = env.action_space.n

    model = SantaFeLSTM(observation_shape, hidden_size, num_actions).to(device)
    target_model = SantaFeLSTM(observation_shape, hidden_size, num_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.SmoothL1Loss()
    prioritized_replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_size)

    epsilon = 1.0
    epsilon_end = 0.1
    total_rewards = []

    for episode in range(30):
        obs, info = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        done = False
        total_reward = 0
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(obs)
                    action = torch.argmax(q_values, dim=1).item()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            prioritized_replay_buffer.add(obs, action, reward, next_obs_tensor, done)
            obs = next_obs_tensor
            total_reward += reward

            if len(prioritized_replay_buffer.buffer) >= batch_size:
                batch, indices, weights = prioritized_replay_buffer.sample(batch_size)
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
                obs_batch = torch.stack([
                    torch.tensor(o, dtype=torch.float32).to(device).squeeze(0)
                    if isinstance(o, np.ndarray) else o.to(device).squeeze(0)
                    for o in obs_batch
                ])
                next_obs_batch = torch.stack([
                    torch.tensor(o, dtype=torch.float32).to(device).squeeze(0)
                    if isinstance(o, np.ndarray) else o.to(device).squeeze(0)
                    for o in next_obs_batch
                ])
                action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
                done_batch = torch.tensor([float(d) for d in done_batch], dtype=torch.float32, device=device)
                q_values = model(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_model(next_obs_batch)
                    max_next_q_values = next_q_values.max(1)[0]
                    target = reward_batch + (1 - done_batch) * gamma * max_next_q_values
                loss = loss_fn(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        total_rewards.append(total_reward)
    avg_reward = np.mean(total_rewards[-10:])
    return avg_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print("Best trial:", study.best_trial.params)
