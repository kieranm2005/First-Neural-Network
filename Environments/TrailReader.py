import os
# Convert text file of coordinates to a list of tuples
def load_trail_coordinates(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    coordinates = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                x, y = map(int, line.split(","))
                coordinates.append((x, y))
            except ValueError:
                continue

    print("Loaded coordinates:", coordinates)
    return coordinates

