import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap():
    positions_dir = os.path.join(os.path.dirname(__file__), '../Data/SantaFeTrail-SNN/Positions')
    size = 32  # Grid size

    # Initialize frequency grid
    freq_grid = np.zeros((size, size), dtype=int)

    # Iterate over all position files
    for fname in os.listdir(positions_dir):
        if fname.endswith('.json'):
            with open(os.path.join(positions_dir, fname), 'r') as f:
                positions = json.load(f)
                for pos in positions:
                    # Ensure pos is a list or tuple of length 2
                    if isinstance(pos, (list, tuple)) and len(pos) == 2:
                        #flip y-coordinate to match grid orientation
                        pos = (pos[0], size - 1 - pos[1])  # Adjust y-coordinate for grid orientation
                        x, y = pos
                        if 0 <= x < size and 0 <= y < size:
                            freq_grid[y, x] += 1  # y is row, x is column

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(freq_grid, cmap='hot', interpolation='nearest', origin='lower')
    cbar = fig.colorbar(im, ax=ax, label='Visit Frequency', fraction=0.046, pad=0.04)
    ax.set_title('Santa Fe Trail Agent Position Heatmap')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.tight_layout()
    return fig