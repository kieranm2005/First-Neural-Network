import matplotlib.pyplot as plt
import numpy as np
import json
import os

def get_recent_stats_files(directory, n=3):
    files = [f for f in os.listdir(directory) if f.startswith('episode_stats') and f.endswith('.json')]
    if not files:
        return []
    files.sort(reverse=True)
    return [os.path.join(directory, f) for f in files[:n]]

def load_rewards_from_json(json_path):
    if not json_path or not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
            return [ep['total_reward'] for ep in data if 'total_reward' in ep]
        except Exception:
            return []

def plot_recent_rnn_runs():
    base = os.path.dirname(os.path.abspath(__file__))
    rnn_dir = os.path.join(base, 'SantaFeTrail-RNN')
    recent_files = get_recent_stats_files(rnn_dir, n=3)
    colors = ['orange', 'red', 'purple']
    plt.figure(figsize=(12, 6))
    for idx, (file, color) in enumerate(zip(recent_files, colors)):
        rewards = load_rewards_from_json(file)
        if rewards:
            subsampled_rewards = rewards[::10]
            plt.plot(np.arange(0, len(subsampled_rewards)*10, 10), subsampled_rewards, label=f'RNN Run {idx+1}: {os.path.basename(file)}', color=color)
    plt.axhline(89, color='black', linestyle='--', linewidth=1, label='Y=89')
    plt.title('3 Most Recent RNN Training Runs')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    plot_recent_rnn_runs()
