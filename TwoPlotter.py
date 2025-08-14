import json
import os
import matplotlib.pyplot as plt

# Base path and folders
base_path = "Data"
folders = ["SantaFeTrail-RNN", "SantaFeTrail-SNN"]
data_dict = {}

# Parameters
MAX_EPISODES = 300
CHUNK_SIZE = 2

# Function to compute average rewards in chunks
def average_rewards(data, chunk_size=1):
    rewards = [entry["total_reward"] for entry in data]
    averaged = [
        sum(rewards[i:i + chunk_size]) / len(rewards[i:i + chunk_size])
        for i in range(0, len(rewards), chunk_size)
    ]
    chunk_ids = list(range(1, len(averaged) + 1))
    return chunk_ids, averaged

# Load and process each model
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    
    # Find the most recent file
    files = [f for f in os.listdir(folder_path) if f.startswith("episode_stats_")]
    if not files:
        print(f"No data files found in {folder_path}")
        continue

    most_recent_file = sorted(files, reverse=True)[0]
    file_path = os.path.join(folder_path, most_recent_file)
    print(f"[{folder}] Loading: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)
        data = data[:MAX_EPISODES]  # truncate to 300 episodes max
        chunks, avg_rewards = average_rewards(data, chunk_size=CHUNK_SIZE)
        data_dict[folder] = (chunks, avg_rewards)

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

for ax, (label, (chunks, avg_rewards)) in zip(axes, data_dict.items()):
    ax.plot(chunks, avg_rewards, linestyle='-', label=label)
    ax.set_title(f"Average Total Reward per 10 Episodes - {label}")
    ax.set_ylabel("Average Reward")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Chunk (2 Episodes Each)")
plt.tight_layout()
plt.show()
