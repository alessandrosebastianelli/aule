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

## What's included

Metrics are organized by family in `aule.metrics`, all importable directly from `aule.metrics`:

- **core**: `rmse`, `mse`, `mae`, `bias`, `pearson_r`, `ssim`, `psnr`, `r2_score`, `mape`, `smape`, `nse`, `kge`, `max_error`, `explained_variance`
- **spectral**: `spectral_error`, `gradient_error`, `psd_radial_error`, `spectral_angle_mapper`
- **climate**: `seasonal_error`, `percentile_error`, `pixelwise_temporal_correlation`, `trend_error`, `extreme_event_duration_error`, `autocorrelation_error`
- **ensemble**: `ensemble_spread`, `crps`, `rank_histogram`, `brier_score`, `spread_skill_ratio`, `crps_skill_score`
- **earth_observation**: `normalized_difference_index`, `index_error`, `change_detection_error`
- **classification**: `iou`, `dice`, `precision_recall_f1`, `confusion_matrix_metrics`, `cohen_kappa` (binary or multi-class, via `average`/`num_classes`)
- **uncertainty**: `picp`, `pit_histogram`

Plots are organized similarly in `aule.plots`:

- **core**: `plot_scatter`, `plot_qq`, `plot_histogram_comparison`, `plot_error_histogram`
- **spatial**: `plot_field_comparison`, `plot_bias_map`, `plot_correlation_map` (optional cartopy basemap via `lat`/`lon`)
- **climate**: `plot_temporal_trend`, `plot_temporal_scatter`
- **ensemble**: `plot_ensemble_spread_map`, `plot_rank_histogram`
- **diagnostics**: `plot_taylor_diagram`, `plot_boxplot_comparison`, `plot_violin_comparison`, `plot_time_series`, `plot_error_map`
- **classification**: `plot_confusion_matrix`, `plot_reliability_diagram`

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

## Notebooks

The `notebooks/` folder contains worked examples for every metric and plot
family, each runnable end-to-end:

1. `01_core_metrics.ipynb`
2. `02_spectral_and_earth_observation_metrics.ipynb`
3. `03_climate_metrics.ipynb`
4. `04_ensemble_and_uncertainty_metrics.ipynb`
5. `05_classification_metrics.ipynb`
6. `06_plots.ipynb`
7. `07_aule_class.ipynb`

## Documentation

The documentation is produced using [pdoc](https://pdoc.dev).

```bash
python build_doc.py
```