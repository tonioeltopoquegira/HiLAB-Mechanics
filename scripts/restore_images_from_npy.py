"""
A utility to restore PNG images from .npy files.

This script is useful if image files were accidentally deleted but the
NumPy array data was preserved. It iterates through a target directory,
finds all .npy files, and saves a corresponding .png image for each.
"""
import argparse
import os
import numpy as np
from PIL import Image

def restore_images(directory: str):
    """
    Finds all .npy files in a directory and saves them as PNG images.

    Args:
        directory (str): The path to the directory to process.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"Scanning for .npy files in '{directory}'...")
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]

    if not npy_files:
        print("No .npy files found.")
        return

    for npy_file in npy_files:
        npy_path = os.path.join(directory, npy_file)
        img_path = os.path.splitext(npy_path)[0] + '.png'

        try:
            # Load the numpy array
            array_data = np.load(npy_path)

            # Ensure the array is in a format suitable for image conversion
            # Assuming the data is float between 0.0 and 1.0
            if array_data.dtype != np.uint8:
                array_data = (array_data * 255).astype(np.uint8)
            
            # Handle different array shapes. We expect HxW or HxWx1/3
            if len(array_data.shape) == 3 and array_data.shape[2] == 1:
                array_data = array_data.squeeze(axis=2) # Grayscale from HxWx1

            # Create and save the image
            img = Image.fromarray(array_data)
            img.save(img_path)
            print(f"Restored '{img_path}'")

        except Exception as e:
            print(f"Could not process '{npy_file}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Restore PNG images from .npy files in a specified directory."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The path to the directory containing the .npy files.",
    )
    args = parser.parse_args()
    restore_images(args.directory)

if __name__ == "__main__":
    main()
