# aule

**aule** provides validation metrics and plots for machine learning model outputs,
with a focus on earth observation and climate science use cases (gridded fields,
ensembles, multi-temporal data).

Named after Aulë, the Vala of craft in Tolkien's mythology — the one who forges
and evaluates the work of his own hands.

## Supported input shapes

Every function accepts numpy arrays in one of four shapes:

- `(batch, H, W, C)`
- `(batch, H, W, C, T)`
- `(H, W, C)`
- `(H, W, C, T)`

When an array is 4D, pass `data_format="bhwc"` (default) or `data_format="hwct"`
to disambiguate, since the two shapes cannot be told apart from shape alone.

## Installation

```bash
pip install aule

# with cartopy support for geographic basemaps
pip install aule[geo]
```

## Quick example

```python
import numpy as np
from aule.metrics import rmse, mae, pearson_r, ssim
from aule.plots import plot_field_comparison, plot_scatter

gt   = np.random.rand(64, 64, 1)
pred = gt + np.random.normal(0, 0.1, gt.shape)

print(rmse(gt, pred))
print(pearson_r(gt, pred))

fig, axes = plot_field_comparison(gt, pred)
fig, ax = plot_scatter(gt, pred, save_path="scatter.png")
```

## Object-oriented usage

Every metric and plot is also available as a method on the `aule` class,
which binds `y_true`/`y_pred` (and optionally `data_format`/`ignore_nan`)
once. New functions added to `aule.metrics` or `aule.plots` are picked up
automatically, no extra wiring needed.

```python
from aule import aule

v = aule(gt, pred)
print(v.rmse())
print(v.pearson_r())
fig, ax = v.plot_scatter(save_path="scatter.png")
```

## Documentation

The documentation is produced using [pdoc](https://pdoc.dev).

```bash
python build_doc.py
```