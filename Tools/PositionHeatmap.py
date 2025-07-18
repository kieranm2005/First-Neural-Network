import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_position_heatmap_from_flat(positions_dir, output_path=None, grid_shape=(32, 32)):
    """
    Plot a heatmap from flattened position arrays stored in JSON files.
    Args:
        positions_dir (str): Directory containing agent_positions_ep*.json files
        output_path (str, optional): Path to save the heatmap image. If None, displays the plot.
        grid_shape (tuple): Shape of the 2D grid (default is 32x32)
    """
    heatmap = np.zeros(grid_shape, dtype=float)

    for fname in os.listdir(positions_dir):
        if fname.startswith('agent_positions_ep') and fname.endswith('.json'):
            with open(os.path.join(positions_dir, fname), 'r') as f:
                flat_data = json.load(f)
                # Flatten list of lists of single floats into 1D list
                flat_array = np.array([x[0] if isinstance(x, list) else x for x in flat_data])
                if flat_array.size != np.prod(grid_shape):
                    print(f"Skipping {fname}: unexpected size {flat_array.size}")
                    continue
                reshaped = flat_array.reshape(grid_shape)
                heatmap += reshaped

    if heatmap.sum() == 0:
        raise ValueError("No valid data found to plot.")

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='hot', origin='lower')
    plt.colorbar(label='Visit Frequency')
    plt.title('Agent Position Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
