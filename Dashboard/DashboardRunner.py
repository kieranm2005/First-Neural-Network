"""
DashboardRunner.py

This script provides a NiceGUI-based dashboard for visualizing and comparing the performance statistics
of different neural network types (RNN, CNN, SNN, FSNN) on the Santa Fe Trail problem. It loads the latest
episode statistics from JSON files, displays reward trends using matplotlib, and summarizes best, worst,
and average rewards for each network type. Usage: Run this script to launch the dashboard web interface.
"""

from nicegui import ui
import numpy as np
import plotly.graph_objs as go
import os, json

ui.markdown('# Neural Network Dashboard')



NN_TYPES = ['RNN', 'CNN', 'SNN', 'FSNN']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
stats_dirs = {
    'RNN': os.path.join(BASE_DIR, '../Data/SantaFeTrail-RNN'),
    'CNN': os.path.join(BASE_DIR, '../Data/SantaFeTrail-CNN'),
    'SNN': os.path.join(BASE_DIR, '../Data/SantaFeTrail-SNN'),
    'FSNN': os.path.join(BASE_DIR, '../Data/SantaFeTrail-FSNN'),
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

def get_stats_files_with_dates(nn_type):
    dir_path = stats_dirs.get(nn_type)
    if not dir_path or not os.path.isdir(dir_path):
        return []
    files = [entry.name for entry in os.scandir(dir_path) if entry.is_file() and entry.name.startswith('episode_stats_') and entry.name.endswith('.json')]
    from datetime import datetime
    file_info = []
    for f in files:
        ts = f.replace('episode_stats_', '').replace('.json', '')
        try:
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        except ValueError:
            try:
                dt = datetime.strptime(ts, "%Y%m%d")
            except ValueError:
                dt = None
        file_info.append((f, dt))
    # Sort by datetime descending (most recent first)
    file_info = sorted(file_info, key=lambda x: x[1] if x[1] else datetime.min, reverse=True)
    return file_info

# --- UI Setup ---
plotly_fig = go.Figure()
plotly_fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Reward'))
plotly_plot = ui.plotly(plotly_fig).classes('w-full h-64')

stats_md = ui.markdown('### Best Reward: Worst Reward: Average Reward:')

# State for menu
selected_nn = {'type': NN_TYPES[0]}
selected_file = {'name': None}

file_dropdown = None

def update_dashboard(nn_type, file_name=None):
    # If no file_name, use most recent
    files = get_stats_files_with_dates(nn_type)
    if not files:
        rewards, stats = [], None
        stats_text = "### Best Reward: N/A  Worst Reward: N/A  Average Reward: N/A"
        plotly_fig.data = []
        plotly_fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Reward'))
        plotly_fig.update_layout(title=f'{nn_type} Rewards', xaxis_title='Episode', yaxis_title='Reward', showlegend=True)
        plotly_plot.update()
        stats_md.set_content(stats_text)
        return
    if file_name is None:
        file_name = files[0][0]
    path = os.path.join(stats_dirs[nn_type], file_name)
    with open(path) as f:
        stats = json.load(f)
    rewards = [ep['total_reward'] for ep in stats if 'total_reward' in ep]
    if rewards:
        x = list(range(1, len(rewards) + 1))
        plotly_fig.data = []
        plotly_fig.add_trace(go.Scatter(x=x, y=rewards, mode='lines', name='Reward'))
        plotly_fig.update_layout(
            title=f'{nn_type} Rewards',
            xaxis_title='Episode',
            yaxis_title='Reward',
            showlegend=True
        )
        best = np.max(rewards)
        worst = np.min(rewards)
        avg = np.mean(rewards)
        stats_text = f"### Best Reward: {best:.2f}  Worst Reward: {worst:.2f}  Average Reward: {avg:.2f}"
    else:
        plotly_fig.data = []
        plotly_fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Reward'))
        plotly_fig.update_layout(title=f'{nn_type} Rewards', xaxis_title='Episode', yaxis_title='Reward', showlegend=True)
        stats_text = "### Best Reward: N/A  Worst Reward: N/A  Average Reward: N/A"
    plotly_plot.update()
    stats_md.set_content(stats_text)
    selected_file['name'] = file_name
    # Update dropdown menu if needed
    if file_dropdown:
        file_dropdown.options = [(f"{dt.strftime('%Y-%m-%d %H:%M:%S') if dt else f}", f) for f, dt in files]
        file_dropdown.value = file_name

def on_nn_change(nn_type):
    selected_nn['type'] = nn_type
    files = get_stats_files_with_dates(nn_type)
    if files:
        update_dashboard(nn_type, files[0][0])
    else:
        update_dashboard(nn_type, None)
    # Update dropdown menu
    if file_dropdown:
        file_dropdown.options = [(f"{dt.strftime('%Y-%m-%d %H:%M:%S') if dt else f}", f) for f, dt in files]
        file_dropdown.value = files[0][0] if files else None

def on_file_change(file_name):
    update_dashboard(selected_nn['type'], file_name)

with ui.button_group().props('rounded glossy'):
    for nn in NN_TYPES:
        ui.button(nn, on_click=lambda nn=nn: on_nn_change(nn)).props('push')

# Dropdown for stats file selection
files = get_stats_files_with_dates(NN_TYPES[0])
file_options = [(f"{dt.strftime('%Y-%m-%d %H:%M:%S') if dt else f}", f) for f, dt in files]
option_values = [v for _, v in file_options]
# Ensure file_dropdown_value is valid before setting it
file_dropdown_value = file_options[0][1] if file_options else None

# Update dropdown initialization to handle empty options
file_dropdown = ui.select(
    options=file_options,
    value=file_dropdown_value,
    on_change=lambda e: on_file_change(e.value)
).props('outlined dense')

update_dashboard(NN_TYPES[0], file_dropdown_value)
ui.run()

