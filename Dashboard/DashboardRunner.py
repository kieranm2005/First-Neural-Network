"""
DashboardRunner.py

This script provides a NiceGUI-based dashboard for visualizing and comparing the performance statistics
of different neural network types (RNN, CNN, SNN, FSNN) on the Santa Fe Trail problem. It loads the latest
episode statistics from JSON files, displays reward trends using matplotlib, and summarizes best, worst,
and average rewards for each network type. Usage: Run this script to launch the dashboard web interface.
"""

from nicegui import ui
import numpy as np
from matplotlib import pyplot as plt
import os, json

ui.markdown('# Neural Network Dashboard')



NN_TYPES = ['RNN', 'CNN', 'SNN', 'FSNN']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
stats_dirs = {
    'RNN': os.path.join(BASE_DIR, '../SantaFeTrail-RNN'),
    'CNN': os.path.join(BASE_DIR, '../SantaFeTrail-CNN'),
    'SNN': os.path.join(BASE_DIR, '../SantaFeTrail-SNN'),
    'FSNN': os.path.join(BASE_DIR, '../SantaFeTrail-FSNN'),
}

def get_latest_stats_file(nn_type):
    dir_path = stats_dirs.get(nn_type)
    if not dir_path or not os.path.isdir(dir_path):
        return None
    files = [entry.name for entry in os.scandir(dir_path) if entry.is_file() and entry.name.startswith('episode_stats_') and entry.name.endswith('.json')]
    if not files:
        return None
    # Sort by timestamp in filename (YYYYMMDD_HHMMSS) as integer for correct order
    from datetime import datetime
    def extract_timestamp(f):
        # Expecting episode_stats_YYYYMMDD_HHMMSS.json
        ts = f.replace('episode_stats_', '').replace('.json', '')
        try:
            # Try parsing full timestamp
            return datetime.strptime(ts, "%Y%m%d_%H%M%S")
        except ValueError:
            try:
                # Try parsing only date part
                return datetime.strptime(ts, "%Y%m%d")
            except ValueError:
                return datetime.min
    files = sorted(files, key=extract_timestamp, reverse=True)
    chosen = os.path.join(dir_path, files[0]) if files else None
    print(f"[DEBUG] {nn_type} latest stats file: {chosen}")
    return chosen



def load_stats(nn_type):
    path = get_latest_stats_file(nn_type)
    if not path or not os.path.exists(path):
        print(f"[DEBUG] No stats file found for {nn_type}")
        return [], None
    with open(path) as f:
        stats = json.load(f)
    rewards = []
    for idx, ep in enumerate(stats):
        if 'total_reward' in ep:
            rewards.append(ep['total_reward'])
        else:
            print(f"[WARNING] Episode at index {idx} missing 'total_reward' key in {nn_type} stats.")
    print(f"[DEBUG] Loaded {len(rewards)} rewards for {nn_type}. Sample: {rewards[:3] if rewards else 'None'}")
    return rewards, stats



# --- UI Setup ---
with ui.pyplot(figsize=(5, 3)) as plot:
    plt.plot([], [])

stats_md = ui.markdown('### Best Reward: Worst Reward: Average Reward:')

def update_dashboard(nn_type):
    rewards, stats = load_stats(nn_type)
    with plot:
        plt.clf()  # Clear the current figure
        ax = plt.gca()  # Get the current axes
        if rewards:
            ax.plot(rewards, label='Reward')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title(f'{nn_type} Rewards')
            ax.legend()
            best = np.max(rewards)
            worst = np.min(rewards)
            avg = np.mean(rewards)
            stats_text = f"### Best Reward: {best:.2f}  Worst Reward: {worst:.2f}  Average Reward: {avg:.2f}"
        else:
            stats_text = "### Best Reward: N/A  Worst Reward: N/A  Average Reward: N/A"
        stats_md.set_content(stats_text)

def create_nn_button(nn):
    ui.button(nn, on_click=lambda: update_dashboard(nn)).props('push')

with ui.button_group().props('rounded glossy'):
    for nn in NN_TYPES:
        create_nn_button(nn)

with ui.button_group().props('rounded glossy'):
    # Find the first NN type with available stats
    default_nn_type = None
    for nn in NN_TYPES:
        rewards, stats = load_stats(nn)
        if rewards:
            default_nn_type = nn
            break
    if default_nn_type is None:
        default_nn_type = NN_TYPES[0]  # fallback if none found

    update_dashboard(default_nn_type)

ui.run()
update_dashboard(NN_TYPES[0])

ui.run()

