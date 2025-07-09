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
    files = [f for f in os.listdir(dir_path) if f.startswith('episode_stats_') and f.endswith('.json')]
    if not files:
        return None
    # Sort by timestamp in filename (YYYYMMDD_HHMMSS) as integer for correct order
    def extract_timestamp(f):
        # Expecting episode_stats_YYYYMMDD_HHMMSS.json
        ts = f.replace('episode_stats_', '').replace('.json', '')
        parts = ts.split('_')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0] + parts[1])
        elif parts[0].isdigit():
            return int(parts[0])
        else:
            return 0
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
    rewards = [ep['total_reward'] for ep in stats if 'total_reward' in ep]
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
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            stats_text = "### No stats available."
        plot.update()
    stats_md.set_content(stats_text)

with ui.button_group().props('rounded glossy'):
    for nn in NN_TYPES:
        ui.button(nn, on_click=lambda nn=nn: update_dashboard(nn)).props('push')

update_dashboard(NN_TYPES[0])

ui.run()

