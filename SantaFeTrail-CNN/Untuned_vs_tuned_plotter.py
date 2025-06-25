import matplotlib.pyplot as plt
import json
from Plotter import get_episodes, get_rewards

# Load Plotter.py data
plotter_episodes = get_episodes()
plotter_rewards = get_rewards()

# Load episode_stats.json data
with open("SantaFeTrail-CNN/episode_stats.json", "r") as f:
    stats = json.load(f)
json_episodes = [entry["episode"] for entry in stats]
json_rewards = [entry["total_reward"] for entry in stats]

# Subsample both datasets for clearer plotting
subsample_rate = 10
plotter_episodes_sub = plotter_episodes[::subsample_rate]
plotter_rewards_sub = plotter_rewards[::subsample_rate]
json_episodes_sub = json_episodes[::subsample_rate]
json_rewards_sub = json_rewards[::subsample_rate]

# Plot both on the same figure (subsampled)
plt.figure(figsize=(12, 6))
plt.plot(plotter_episodes_sub, plotter_rewards_sub, label="Default hyperparameters", alpha=0.7)
plt.plot(json_episodes_sub, json_rewards_sub, label="Tuned hyperparameters", alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode: Untuned vs Tuned Hyperparameters (Subsampled)")
plt.legend()
plt.show()
