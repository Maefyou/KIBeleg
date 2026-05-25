# Analysis

This directory is for experiment analysis and comparison plots.

## Current output

- `layers_heads_heatmap.png`: loss heatmap for the model sweep across layers and heads.
- `layers_heads_heatmap.json`: structured summary data used to create the heatmap.

## Regenerate the heatmap

```bash
.venv/bin/python analysis/generate_heatmap.py --run-root runs
```

You can also choose a different metric:

```bash
.venv/bin/python analysis/generate_heatmap.py --run-root runs --metric last_window_loss_avg
```
