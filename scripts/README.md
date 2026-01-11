Generate topology designs

This folder contains a helper script to run the repository's inverse-design routines
and save results for later training of learned representations.

Usage
-----
Run the generator script (example):

```bash
python scripts/generate_designs.py --output_dir outputs/designs --opt_steps 100
```

Options
-------
- `--problem`: problem key from `problems.PROBLEMS_BY_NAME` (default: `mbb_beam_192x64_0.4`).
- `--opt_steps`: number of optimization iterations (default: 100).
- `--volfrac`, `--penal_start`, `--penal_end`, `--penal_power`: override TO params.

Outputs
-------
Each run creates a timestamped directory under `outputs/designs/` with:

- `config.json` — the JSON-serialized TO arguments used.
- `mma.nc` — xarray Dataset with MMA optimization trace (if successful).
- `mma_final.png` — final rendered image of MMA design.
- `pixel_lbfgs.nc` — xarray Dataset with Pixel-LBFGS trace (if successful).
- `pixel_lbfgs_final.png` — final rendered image of Pixel-LBFGS design.
