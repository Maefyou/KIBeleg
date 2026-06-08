# Sweep batch: layers 5–8 × heads 5–8

Architecture sweep motivated by the earlier `runs/` analysis (layers/heads 1–4),
which indicated that **more heads and more layers help**.

## Data / training settings

- Coordinate range: points and circle in **[-0.5, 0.5]²**
- Circle radius: `U(0.1, 0.25)` (min fixed at absolute 0.1, max = `0.5 · plane_width/2`, always fully inside the field)
- `num_points = 200`, `epochs = 15000`, `batch_size = 64`, `lr = 1e-4`
- Noise on (`std = 0.05`) and clutter on (`fraction = 0.1`)
- Base `d_model = 120`
- Primary metric: **min loss over the last 100 epochs** (`last_window_loss_min`)

## ⚠️ d_model exception for heads = 7

`d_model` must be divisible by the number of attention heads (multi-head attention
splits `d_model` into `nhead` equal slices) and must be even (positional encoding).
`120` satisfies this for heads 5, 6, 8 (`120/5=24`, `120/6=20`, `120/8=15`) but **not 7**
(`120/7` is not an integer). There is no small width divisible by all of 5, 6, 7, 8
(the LCM is 840, far too wide/slow).

**Decision:** keep `d_model = 120` for heads 5, 6, 8, and use **`d_model = 126`** (`= 7·18`,
the nearest even multiple of 7 to 120) **only for heads = 7**. These models are therefore
~5% wider than the rest. This is handled automatically by `resolve_d_model()` in
`sweep_models.py`, and is reflected in the run directory names:

| heads | d_model | run dir suffix |
|-------|---------|----------------|
| 5 | 120 | `..._dmodel_120` |
| 6 | 120 | `..._dmodel_120` |
| 7 | **126** | `..._dmodel_126` |
| 8 | 120 | `..._dmodel_120` |

When interpreting the heatmap, keep in mind the heads = 7 column was trained at the
slightly larger width 126 rather than 120.

## Reproduce / resume

```bash
.venv/bin/python sweep_models.py \
  --layers-min 5 --layers-max 8 --heads-min 5 --heads-max 8 \
  --d-model 120 --num-points 200 --epochs 15000 --batch-size 64 --lr 0.0001 \
  --noise --clutter --run-root runs_5to8 --skip-existing
```

`--skip-existing` skips only **completed** runs (those with a `metrics.json`), so the
command is safe to re-run to resume after an interruption.
