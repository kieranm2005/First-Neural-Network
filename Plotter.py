import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Utility to find the most recent stats file in a directory
def get_latest_stats_file(directory):
    files = [f for f in os.listdir(directory) if f.startswith('episode_stats') and f.endswith('.json')]
    if not files:
        return None
    files.sort(reverse=True)
    return os.path.join(directory, files[0])

# Load rewards from a stats file
def load_rewards_from_json(json_path):
    if not json_path or not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
            return [ep['total_reward'] for ep in data if 'total_reward' in ep]
        except Exception:
            return []

# Plot CNN, RNN, and SNN training results
def plot_training_results(cnn_rewards, rnn_rewards, snn_rewards, title="Training Results"):
    plt.figure(figsize=(12, 6))
    plt.plot(cnn_rewards, label='CNN Rewards', color='blue')
    plt.plot(rnn_rewards, label='RNN Rewards', color='orange')
    plt.plot(snn_rewards, label='SNN Rewards', color='green')
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.grid()
    plt.show()

def plot_training_results_from_files():
    base = os.path.dirname(os.path.abspath(__file__))
    cnn_file = get_latest_stats_file(os.path.join(base, 'SantaFeTrail-CNN'))
    rnn_file = get_latest_stats_file(os.path.join(base, 'SantaFeTrail-RNN'))
    snn_file = get_latest_stats_file(os.path.join(base, 'SantaFeTrail-SNN'))
    cnn_rewards = load_rewards_from_json(cnn_file)[::10]
    rnn_rewards = load_rewards_from_json(rnn_file)[::10]
    snn_rewards = load_rewards_from_json(snn_file)[::10]
    plt.figure(figsize=(12, 6))
    if cnn_rewards:
        plt.plot(np.arange(0, len(cnn_rewards)*10, 10), cnn_rewards, label='CNN Rewards', color='blue')
    if rnn_rewards:
        plt.plot(np.arange(0, len(rnn_rewards)*10, 10), rnn_rewards, label='RNN Rewards', color='orange')
    if snn_rewards:
        plt.plot(np.arange(0, len(snn_rewards)*10, 10), snn_rewards, label='SNN Rewards', color='green')
    plt.axhline(89, color='red', linestyle='--', linewidth=1, label='Y=89')
    plt.title('Training Results (Most Recent, Subsampled x10)')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    plot_training_results_from_files()
    # Example usage with hardcoded values
    # plot_training_results([1, 2, 3], [2, 3, 4], [3, 4, 5], title="Example Training Results")
