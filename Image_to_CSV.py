#Converting 28x28 images to CSV format for Keras model training
import numpy as np
import os
import pandas as pd
from PIL import Image

def convert_images_to_csv(image_folder, csv_file):
    """
    Convert images in a folder to a CSV file suitable for Keras model training.
    
    Parameters:
    image_folder (str): Path to the folder containing images.
    csv_file (str): Path where the CSV file will be saved.
    """
    data = []
    
    # Iterate through all files in the image folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28 pixels
            img_array = np.array(img).flatten()  # Flatten the image to a 1D array
            
            # Determine label as alphabet index (A=1, B=2, ..., Z=26)
            letter = filename.split('.')[0].upper()
            if len(letter) == 1 and 'A' <= letter <= 'Z':
                label = ord(letter) - ord('A') + 1
            else:
                continue  # skip files not named as a single letter
            # Insert label at the start, shift pixels right
            row = np.insert(img_array, 0, label)
            data.append(row)
    
    # Create a DataFrame and save it as a CSV file
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False, header=False)  # Save without index and header

convert_images_to_csv('/u/kieranm/Documents/Dataset/Initials', '/u/kieranm/Documents/Dataset/initials.csv')