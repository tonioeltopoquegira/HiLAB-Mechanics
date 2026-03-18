"""
Recreates grayscale PNG images from design optimization NumPy arrays.

This script is tailored to the project's output directory structure. It takes
an experiment name, finds the NumPy arrays (`.npy`) containing designs from
different optimizers (MMA, OC, Pixel L-BFGS), and saves each design as a
separate grayscale PNG image in the corresponding 'images' folder.

This is useful for regenerating visualizations if the image files were
deleted or for creating them for the first time from raw array data.

Example:
    python -m neural_structural_optimization.recreate_design_images mbb_beam_384x64_aug
"""
import argparse
import os
import numpy as np
from PIL import Image

def recreate_images_for_experiment(experiment_name: str):
    """
    Finds design arrays for an experiment and saves each design as a PNG.

    Args:
        experiment_name (str): The name of the experiment folder in
                               `outputs/designs/`.
    """
    base_path = "outputs/designs"
    experiment_path = os.path.join(base_path, experiment_name)
    arrays_path = os.path.join(experiment_path, "arrays")
    images_path = os.path.join(experiment_path, "images")

    if not os.path.isdir(arrays_path):
        print(f"Error: Arrays directory not found at '{arrays_path}'")
        print("Please make sure you have run an optimization for this experiment.")
        return

    # Ensure the output directory exists
    os.makedirs(images_path, exist_ok=True)
    print(f"Output images will be saved in: '{images_path}'")

    # Find all .npy files in the arrays directory
    try:
        npy_files = [f for f in os.listdir(arrays_path) if f.endswith('.npy')]
    except FileNotFoundError:
        print(f"Error: Could not list files in '{arrays_path}'. Does it exist?")
        return

    if not npy_files:
        print(f"No .npy files found in '{arrays_path}'.")
        return

    for npy_file in npy_files:
        '''if "binary" in npy_file:
            print(f"Skipping binary file: '{npy_file}'")
            continue'''
        npy_path = os.path.join(arrays_path, npy_file)
        base_name = os.path.splitext(npy_file)[0] # e.g., 'mma_designs'

        try:
            # Load the numpy array. It might contain multiple designs.
            designs = np.load(npy_path)

            # If the array has fewer than 3 dimensions, it's likely a single image
            # or a 1D array. We'll wrap it to handle it uniformly.
            if designs.ndim < 3:
                designs = np.expand_dims(designs, axis=0)

            print(f"Processing '{npy_file}' with {len(designs)} designs...")

            for i, design_array in enumerate(designs):
                # Define a unique name for each image
                img_name = f"{base_name}_{i}.png"
                img_path = os.path.join(images_path, img_name)

                # Invert the array for correct black/white display
                # Where material=1.0, it should be black (0).
                inverted_array = 1.0 - design_array

                if np.all(design_array == design_array.flat[0]):
                    print(f"  -> Skipping uniform design at index {i} (all black or all white).")
                    continue

                # Convert from float (0.0 to 1.0) to uint8 (0 to 255)
                if inverted_array.dtype != np.uint8:
                    image_array = (inverted_array * 255).astype(np.uint8)
                else:
                    # This case is unlikely but safe to handle
                    image_array = inverted_array

                # Create and save the grayscale image
                img = Image.fromarray(image_array, 'L')
                img.save(img_path)
                print(f"  -> Saved '{img_path}'")

        except Exception as e:
            print(f"Could not process '{npy_file}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Recreate grayscale PNGs from .npy design arrays for a given experiment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        help="The name of the experiment directory under outputs/designs/.\n"
             "Example: mbb_beam_384x64_aug"
    )
    args = parser.parse_args()
    recreate_images_for_experiment(args.experiment_name)

if __name__ == "__main__":
    main()
