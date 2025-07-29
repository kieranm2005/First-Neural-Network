from nicegui import ui
import numpy as np
import matplotlib
import sys
import os
import json

''' TO DO:
- Graphs should be next to each other, not below
- Top row: toroidal heatmap and reward graph
- Bottom row: non-toroidal heatmap and reward graph
- Add flag to data collection to indicate toroidal vs non-toroidal
- Add toggle for showing trail over top of heatmap'''


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Tools')))
from HeatmapPlotter import plot_heatmap

def main():
    ui.label('Santa Fe Trail Dashboard').style('font-size: 2rem; font-weight: bold; margin-bottom: 1rem;')

    with ui.card():

        # Dropdown menu for model selection
        model_options = ['SNN', 'RNN']
        selected_model = ui.select(model_options, value='SNN', label='Select Model').classes('w-48')

        # Toggle for showing trail over heatmap
        show_trail = ui.checkbox('Show Trail Over Heatmap', value=False)

    ui.separator()

    # Reactive heatmap rendering based on dropdown selection
    def render_heatmap():
        with ui.matplotlib(figsize=(8, 8)).figure as fig:
            # Dynamically set positions_dir based on dropdown
            model_type = selected_model.value
            positions_dir = os.path.join(os.path.dirname(__file__), f'../Data/SantaFeTrail-{model_type}/Positions')
            size = 32
            freq_grid = np.zeros((size, size), dtype=int)
            if os.path.exists(positions_dir):
                for fname in os.listdir(positions_dir):
                    if fname.endswith('.json'):
                        with open(os.path.join(positions_dir, fname), 'r') as f:
                            positions = json.load(f)
                            for pos in positions:
                                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                                    pos = (pos[0], size - 1 - pos[1])
                                    x, y = pos
                                    if 0 <= x < size and 0 <= y < size:
                                        freq_grid[y, x] += 1
            ax = fig.gca()
            im = ax.imshow(freq_grid, cmap='hot', interpolation='nearest', origin='lower')
            fig.colorbar(im, ax=ax, label='Visit Frequency', fraction=0.046, pad=0.04)
            ax.set_title(f'Santa Fe Trail Agent Position Heatmap ({model_type})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            fig.tight_layout()

    def render_trail_coordinates(): #Render trail coordinates over heatmap
        with ui.matplotlib(figsize=(8, 8)).figure as fig:
            model_type = selected_model.value
            trail_file = os.path.join(os.path.dirname(__file__), f'../Data/SantaFeTrail-{model_type}/trail_coordinates.json')
            if os.path.exists(trail_file):
                with open(trail_file, 'r') as f:
                    trail_coords = json.load(f)
                    print(f'Trail coordinates loaded: {len(trail_coords)} points')
                ax = fig.gca()
                ax.plot(*zip(*trail_coords), marker='o', color='blue', markersize=2, label='Trail Coordinates')
                ax.set_title(f'Trail Coordinates ({model_type})')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.legend()
                fig.tight_layout()
            else: # If no trail coordinates found, show a message
                ui.label('No trail coordinates found for the selected model.').style('color: red;')

    def render_reward_graph(): #Render rewards per episode
        with ui.matplotlib(figsize=(6, 4)).figure as fig:
            model_type = selected_model.value
            stats_dir = os.path.join(os.path.dirname(__file__), f'../Data/SantaFeTrail-{model_type}')
            # Find the latest episode_stats file
            reward_data = []
            episode_data = []
            if os.path.exists(stats_dir):
                stat_files = [f for f in os.listdir(stats_dir) if f.startswith('episode_stats_') and f.endswith('.json')]
                if stat_files:
                    stat_files.sort(reverse=True)
                    stats_path = os.path.join(stats_dir, stat_files[0])
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                        for entry in stats:
                            if 'episode' in entry and 'total_reward' in entry:
                                episode_data.append(entry['episode'])
                                reward_data.append(entry['total_reward'])
            ax = fig.gca()
            ax.plot(episode_data, reward_data, marker='o', linestyle='-', color='blue')
            ax.set_title(f'Reward per Episode ({model_type})')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            fig.tight_layout()
    # Layout for 4 graphs: heatmap and reward side by side, others below
    with ui.column().classes('w-full'):
        with ui.row().classes('w-full justify-between'):
            heatmap_slot = ui.element().classes('w-1/2')
            reward_slot = ui.element().classes('w-1/2')

        def render_all():
            heatmap_slot.clear()
            reward_slot.clear()
            render_heatmap()
            render_reward_graph()
            render_trail_coordinates()

        def render_heatmap():
            with heatmap_slot:
                with ui.matplotlib(figsize=(8, 8)).figure as fig:
                    model_type = selected_model.value
                    positions_dir = os.path.join(os.path.dirname(__file__), f'../Data/SantaFeTrail-{model_type}/Positions')
                    size = 32
                    freq_grid = np.zeros((size, size), dtype=int)
                    if os.path.exists(positions_dir):
                        for fname in os.listdir(positions_dir):
                            if fname.endswith('.json'):
                                with open(os.path.join(positions_dir, fname), 'r') as f:
                                    positions = json.load(f)
                                    for pos in positions:
                                        if isinstance(pos, (list, tuple)) and len(pos) == 2:
                                            pos = (pos[0], size - 1 - pos[1])
                                            x, y = pos
                                            if 0 <= x < size and 0 <= y < size:
                                                freq_grid[y, x] += 1
                    ax = fig.gca()
                    im = ax.imshow(freq_grid, cmap='hot', interpolation='nearest', origin='lower')
                    fig.colorbar(im, ax=ax, label='Visit Frequency', fraction=0.046, pad=0.04)
                    # Optionally overlay trail coordinates if checkbox is checked
                    if show_trail.value:
                        trail_file = os.path.join(os.path.dirname(__file__), f'../Data/SantaFeTrail-{model_type}/trail_coordinates.json')
                        if os.path.exists(trail_file):
                            with open(trail_file, 'r') as f:
                                trail_coords = json.load(f)
                            ax.plot(*zip(*trail_coords), marker='o', color='blue', markersize=2, label='Trail Coordinates')
                            ax.legend()
                    ax.set_title(f'Santa Fe Trail Agent Position Heatmap ({model_type})')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    fig.tight_layout()

        def render_reward_graph():
            with reward_slot:
                with ui.matplotlib(figsize=(6, 4)).figure as fig:
                    model_type = selected_model.value
                    stats_dir = os.path.join(os.path.dirname(__file__), f'../Data/SantaFeTrail-{model_type}')
                    # Find the latest episode_stats file
                    reward_data = []
                    episode_data = []
                    if os.path.exists(stats_dir):
                        stat_files = [f for f in os.listdir(stats_dir) if f.startswith('episode_stats_') and f.endswith('.json')]
                        if stat_files:
                            stat_files.sort(reverse=True)
                            stats_path = os.path.join(stats_dir, stat_files[0])
                            with open(stats_path, 'r') as f:
                                stats = json.load(f)
                                for entry in stats:
                                    if 'episode' in entry and 'total_reward' in entry:
                                        episode_data.append(entry['episode'])
                                        reward_data.append(entry['total_reward'])
                    ax = fig.gca()
                    ax.plot(episode_data, reward_data, marker='o', linestyle='-', color='blue')
                    ax.set_title(f'Reward per Episode ({model_type})')
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Total Reward')
                    fig.tight_layout()

        render_all()
        def update_all(e=None):
            render_all()
        selected_model.on('update:model-value', update_all)
        show_trail.on('update:value', update_all)

    ui.run()

if __name__ in {"__main__", "__mp_main__"}:
    main()
