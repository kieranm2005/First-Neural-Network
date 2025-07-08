'''Generate trail environments by visually selecting squares on a grid'''
# 1. Prompt user for grid size
# 2. Create a grid of squares
# 3. Allow user to select squares to form a trail, clicking to toggle squares
# 4. Save the selected squares as an array of coordinates to a text file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from matplotlib.widgets import Cursor
import os

class TrailMaker:
    def __init__(self, grid_size=32, trail_name="trail"):
        self.grid_size = grid_size
        self.trail_name = trail_name
        self.selected_squares = set()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, grid_size)
        self.ax.set_ylim(0, grid_size)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('white')

        # Draw all grid squares as light gray rectangles
        self.rect_map = {}
        for x in range(grid_size):
            for y in range(grid_size):
                rect = Rectangle((x, y), 1, 1, facecolor='#f0f0f0', edgecolor='gray', linewidth=0.5)
                self.ax.add_patch(rect)
                self.rect_map[(x, y)] = rect

        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)

        # Connect click event to toggle square selection
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Add a button to save the trail
        ax_save = plt.axes([0.8, 0.01, 0.1, 0.05])
        self.btn_save = Button(ax_save, 'Save Trail')
        self.btn_save.on_clicked(self.save_trail)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x_index = int(event.xdata)
        y_index = int(event.ydata)
        if not (0 <= x_index < self.grid_size and 0 <= y_index < self.grid_size):
            return

        key = (x_index, y_index)
        rect = self.rect_map[key]
        if key in self.selected_squares:
            self.selected_squares.remove(key)
            rect.set_facecolor('#f0f0f0')
        else:
            self.selected_squares.add(key)
            rect.set_facecolor('blue')
        self.fig.canvas.draw_idle()

    def save_trail(self, event):
        trails_dir = os.path.join(os.path.dirname(__file__), "Trails")
        os.makedirs(trails_dir, exist_ok=True)
        filename = os.path.join(trails_dir, f"{self.trail_name}_coordinates.txt")
        print(f"Saving trail coordinates to {filename} ...")
        with open(filename, 'w') as f:
            for square in sorted(self.selected_squares):
                f.write(f"{square[0]}, {square[1]}\n")
        print(f"Trail saved to '{filename}'")

    def show(self):
        plt.show()
        self.fig.canvas.mpl_disconnect(self.cid)
        self.cursor.disconnect_events()
        plt.close(self.fig)
        return self.selected_squares

if __name__ == "__main__":
    # Prompt for trail name and grid size
    trails_dir = os.path.join(os.path.dirname(__file__), "Trails")
    os.makedirs(trails_dir, exist_ok=True)
    while True:
        trail_name = input("Enter a name for the trail: ").strip()
        while not trail_name:
            trail_name = input("Trail name cannot be empty. Enter a name for the trail: ").strip()
        filename = os.path.join(trails_dir, f"{trail_name}_coordinates.txt")
        if os.path.exists(filename):
            overwrite = input(f"WARNING: '{filename}' already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite == "y":
                break
            else:
                print("Please enter a different trail name.")
        else:
            break
    try:
        grid_size = int(input("Enter grid size (default 32): ") or "32")
    except ValueError:
        print("Invalid input, using default grid size 32.")
        grid_size = 32

    trail_maker = TrailMaker(grid_size=grid_size, trail_name=trail_name)
    selected_squares = trail_maker.show()
    print("Selected squares:", selected_squares)