"""Augment saved design images with Gaussian blur and morphological ops.

This script searches an input directory for image files (PNG/JPEG) and for
each image produces:
  - Gaussian-blurred images for sigma in {1.5, 2.0, 2.5}
  - For each blurred image, a morphological erosion and dilation

The output files are saved next to the originals in a mirrored directory
structure under `--output_dir` (default: same as input, under `augmented/`).

Usage:
  python scripts/augment_designs.py --input_dir outputs/designs --output_dir outputs/augmented

"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import scipy.ndimage as ndi


SIGMAS = (0.5, 1.0, 1.5, 2.0)

# Top-level configuration dict (edit this directly; no CLI).
CONFIG = {
    # list of input folders to collect images from
    'input_dirs': ['outputs/designs'],
    # where to write augmented images (mirrors input structure under this dir)
    'output_dir': 'outputs/augmented',
    # gaussian sigmas (pixels)
    'sigmas': list(SIGMAS),
    # scale factor used to convert sigma -> morphology kernel size
    'kernel_scale': 0.5,
    # final binarization threshold for outputs
    'binarize_threshold': 0.5,
}


def find_images(root: Path) -> Iterable[Path]:
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in root.rglob(ext):
            yield p


def load_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.asarray(im).astype(np.float32) / 255.0


def save_gray(arr: np.ndarray, out_path: Path) -> None:
    arr_clipped = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr_clipped).save(out_path)


def morphological_erode(img: np.ndarray, size: int) -> np.ndarray:
    # grayscale erosion
    return ndi.grey_erosion(img, size=(size, size))


def morphological_dilate(img: np.ndarray, size: int) -> np.ndarray:
    # grayscale dilation
    return ndi.grey_dilation(img, size=(size, size))


def augment_image(path: Path, out_dir: Path, sigmas: Sequence[float], kernel_scale: float, bin_thresh: float) -> None:
    """Augment a single image and save binary outputs under out_dir.

    out_dir should already be created by the caller and should mirror the
    intended output location for this input file.
    """
    img = load_gray(path)
    stem = path.stem

    for sigma in sigmas:
        blurred = ndi.gaussian_filter(img, sigma=sigma)

        # Determine morphological kernel size from sigma.
        k = max(1, int(np.ceil(2.0 * sigma * kernel_scale)))

        # Apply morphology on the blurred image, then re-threshold to binary
        eroded = morphological_erode(blurred, k)
        dilated = morphological_dilate(blurred, k)

        # binarize results (final designs must be binary)
        blurred_bin = (blurred >= bin_thresh).astype(np.float32)
        eroded_bin = (eroded >= bin_thresh).astype(np.float32)
        dilated_bin = (dilated >= bin_thresh).astype(np.float32)

        # Save binary outputs
        out_blur = out_dir / f"{stem}_g{sigma:.1f}_binary.png"
        save_gray(blurred_bin, out_blur)

        out_erode = out_dir / f"{stem}_g{sigma:.1f}_erode_binary_k{k}.png"
        save_gray(eroded_bin, out_erode)

        out_dilate = out_dir / f"{stem}_g{sigma:.1f}_dilate_binary_k{k}.png"
        save_gray(dilated_bin, out_dilate)


def main():
    cfg = CONFIG
    input_dirs = [Path(p) for p in cfg.get('input_dirs', [])]
    out_root = Path(cfg.get('output_dir', 'outputs/augmented')).resolve()
    sigmas = list(cfg.get('sigmas', SIGMAS))
    kernel_scale = float(cfg.get('kernel_scale', 1.0))
    bin_thresh = float(cfg.get('binarize_threshold', 0.5))

    all_files = []
    for input_dir in input_dirs:
        input_dir = Path(input_dir).resolve()
        if not input_dir.exists():
            print(f"Warning: input dir does not exist: {input_dir}")
            continue
        files = list(find_images(input_dir))
        all_files.extend([(input_dir, p) for p in files])

    if not all_files:
        print(f"No images found in input_dirs: {input_dirs}")
        return

    print(f"Found {len(all_files)} images across {len(input_dirs)} input dirs; writing augmented images to {out_root}")

    for i, (input_root, fpath) in enumerate(all_files, start=1):
        try:
            rel = fpath.relative_to(input_root)
            out_dir = out_root / input_root.name / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            augment_image(fpath, out_dir, sigmas=sigmas, kernel_scale=kernel_scale, bin_thresh=bin_thresh)
        except Exception as e:
            print(f"Failed to augment {fpath}: {e}")
        if i % 50 == 0:
            print(f"Processed {i}/{len(all_files)}")


if __name__ == '__main__':
    main()
