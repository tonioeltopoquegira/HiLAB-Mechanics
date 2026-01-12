#!/usr/bin/env python3
"""
Generate a sensitivity grid for a decoder's latent space.

Usage examples:
  # use decoder checkpoint and random base latent
  python scripts/latent_variation_grid.py --decoder models/vitvae_decoder_thaw2_latent8.pt --outdir outputs/latent_grid

  # provide explicit base latent (comma-separated 8 values) and schedule
  python scripts/latent_variation_grid.py --decoder models/vitvae_decoder_thaw2_latent8.pt \
      --base 0.5,0.0,-1.0,0,0,0,0,0 --schedule -2,-1,1,2 --mode replace

This will save a `base.png` and `grid.png` in the output folder.
"""
import os
import argparse
import json
from glob import glob

import numpy as np
import torch
from PIL import Image

try:
    from scripts.train_hilab import ViTVAE
except Exception:
    from train_hilab import ViTVAE


def find_latest_decoder(pattern='models/vitvae_decoder_thaw*_latent*.pt'):
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f'No decoder checkpoints found matching {pattern}')
    return files[-1]


def load_decoder(path):
    ck = torch.load(path, map_location='cpu')
    meta = ck.get('meta', {})
    state = ck.get('state', ck)
    latent_dim = int(meta.get('latent_dim', 16))
    model = ViTVAE(latent_dim=latent_dim)
    model_state = model.state_dict()
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
    model.load_state_dict(model_state)
    model.eval()
    return model, latent_dim


def decode_to_rgb(decoder, z: np.ndarray):
    zt = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        recon = decoder.decode(zt)
    recon = recon[0].cpu().numpy()  # (3,H,W)
    rgb = np.clip(recon.transpose(1, 2, 0), 0, 1)
    return rgb


def save_rgb(img_arr, path):
    # img_arr: HxWx3 float in [0,1]
    im = Image.fromarray((img_arr * 255).astype(np.uint8))
    im.save(path)


def make_grid(base_rgb, variations, outpath):
    # variations: list of rows, each row is list of rgb arrays (HxWx3)
    rows = len(variations)
    cols = len(variations[0]) if rows > 0 else 0
    H, W, C = base_rgb.shape
    grid_w = cols * W
    grid_h = rows * H
    grid = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))
    for r in range(rows):
        for c in range(cols):
            arr = variations[r][c]
            im = Image.fromarray((arr * 255).astype(np.uint8))
            grid.paste(im, (c * W, r * H))
    grid.save(outpath)


def parse_latent_list(s: str):
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return np.array([float(x) for x in parts], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--decoder', type=str, default=None, help='Path to decoder checkpoint (.pt). If omitted, picks latest matching models/vitvae_decoder_thaw*_latent*.pt')
    p.add_argument('--base', type=str, default=None, help='Comma-separated base latent vector values (length must match decoder latent_dim). If omitted, samples from N(0,1) with --seed option.')
    p.add_argument('--base_img', type=str, default=None, help='Path to an image file (from your augmented set). If provided, the script will encode it using the encoder to obtain the base latent (uses --full_model or auto-detect).')
    p.add_argument('--full_model', type=str, default=None, help='Path to full ViTVAE state_dict (.pt) that contains encoder weights. If omitted, the script will try to auto-find a matching full model for the decoder checkpoint (models/vitvae_thaw*.pt).')
    p.add_argument('--schedule', nargs='+', default=['-2,-1,1,2'], help='Schedule values: either a single comma-separated string ("-2,-1,1,2") or multiple space-separated values (-2 -1 1 2).')
    p.add_argument('--mode', choices=['add', 'replace'], default='add', help='Whether schedule values are added to base (`add`) or replace the latent entry (`replace`).')
    p.add_argument('--seed', type=int, default=123, help='Random seed when sampling base latent')
    p.add_argument('--outdir', type=str, default='outputs/latent_grid', help='Output directory to save base and grid images')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    decoder_path = args.decoder or find_latest_decoder()
    print('Loading decoder:', decoder_path)
    decoder, latent_dim = load_decoder(decoder_path)
    print('Detected latent_dim =', latent_dim)

    # If user provided a base image, load the full ViTVAE (encoder+decoder) to encode it
    base = None
    if args.base_img is not None:
        # Determine full model path
        full_model_path = args.full_model
        if full_model_path is None:
            # Try to find a recent full model file in models/
            cand = sorted(glob('models/vitvae_thaw2.pt'))
            if cand:
                # pick the newest (last)
                full_model_path = cand[-1]
        if full_model_path is None or not os.path.exists(full_model_path):
            raise FileNotFoundError('Full ViTVAE model state not found. Provide --full_model to enable encoding of base image.')

        print('Loading full model for encoding:', full_model_path)
        # load full model state (state_dict was saved directly in train_hilab)
        full_state = torch.load(full_model_path, map_location='cpu')
        # instantiate full ViTVAE and load state
        full_model = ViTVAE(latent_dim=latent_dim)
        try:
            full_model.load_state_dict(full_state)
        except Exception:
            # attempt to load matching keys (in case saved as {'state_dict': ...})
            if isinstance(full_state, dict) and 'state_dict' in full_state:
                full_model.load_state_dict(full_state['state_dict'])
            else:
                # try to subset matching keys
                cur = full_model.state_dict()
                for k, v in list(full_state.items()):
                    if k in cur and cur[k].shape == v.shape:
                        cur[k] = v
                full_model.load_state_dict(cur)
        full_model.eval()

        # Load and preprocess image
        im = Image.open(args.base_img).convert('RGB')
        im = im.resize((256, 128), resample=Image.BILINEAR)  # PIL size: (width, height)
        arr = np.asarray(im).astype(np.float32) / 255.0  # (H, W, C) => (128,256,3)
        # Convert to tensor NCHW (1,3,128,256)
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        # encode -> mu, logvar
        with torch.no_grad():
            mu, logvar = full_model.encode(img_t)
        base = mu.squeeze(0).cpu().numpy().astype(np.float32)
        print('Encoded base latent from image, shape=', base.shape)
        print(base)

    if base is None:
        if args.base:
            base = parse_latent_list(args.base)
            if base.size != latent_dim:
                raise ValueError(f'Provided base vector length {base.size} != latent_dim {latent_dim}')
        else:
            rng = np.random.RandomState(args.seed)
            base = rng.randn(latent_dim).astype(np.float32)

    # Normalize schedule input: accept either a single comma-separated string or multiple tokens
    schedule_tokens = []
    if isinstance(args.schedule, (list, tuple)):
        for tok in args.schedule:
            schedule_tokens += [t for t in str(tok).split(',') if t.strip()]
    else:
        schedule_tokens = [t for t in str(args.schedule).split(',') if t.strip()]
    schedule_vals = [float(s) for s in schedule_tokens]
    if len(schedule_vals) == 0:
        raise ValueError('Empty schedule provided')

    # Save base image
    base_rgb = decode_to_rgb(decoder, base)
    base_path = os.path.join(args.outdir, 'base.png')
    save_rgb(base_rgb, base_path)
    print('Saved base image to', base_path)

    # Build variations: rows = latent_dim, cols = len(schedule_vals)
    variations = []
    for i in range(latent_dim):
        row = []
        for v in schedule_vals:
            z = base.copy()
            if args.mode == 'add':
                z[i] = z[i] + v
            else:
                z[i] = v
            rgb = decode_to_rgb(decoder, z)
            row.append(rgb)
        variations.append(row)

    grid_path = os.path.join(args.outdir, 'grid.png')
    make_grid(base_rgb, variations, grid_path)
    print('Saved grid to', grid_path)


if __name__ == '__main__':
    main()
