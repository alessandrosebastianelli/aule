# aule

**Validation metrics and plots for machine learning models**, with a focus on earth observation and climate science.

Named after Aulë, the Vala of craft in Tolkien's mythology.

📓 **Extensive usage notebooks are available on [GitHub](https://github.com/alessandrosebastianelli/aule).**

---

## Install

```bash
pip install aule

# optional: geographic basemaps (cartopy)
pip install aule[geo]

# optional: progress bars on heavy loops (tqdm)
pip install aule[progress]
```

---

## Supported input shapes

aule accepts two families of input, both normalised internally to a single canonical `(batch, H, W, C, T)` representation.

### Spatial family

| Shape | Meaning |
|---|---|
| `(H, W, C)` | single spatial field, one channel |
| `(H, W, C, T)` | spatial field with time axis — pass `data_format="hwct"` |
| `(batch, H, W, C)` | batch of spatial fields — pass `data_format="bhwc"` (default) |
| `(batch, H, W, C, T)` | full spatio-temporal batch |

### Series family (pure time series, no spatial extent)

Pass the `axes` keyword to disambiguate:

| Shape | `axes` |
|---|---|
| `(T,)` | `"t"` |
| `(C,)` | `"c"` |
| `(B, T)` | `"bt"` |
| `(C, T)` | `"ct"` |
| `(B, C, T)` | `"bct"` |

Series inputs are promoted to `(B, 1, 1, C, T)` internally, so every generic metric (RMSE, MAE, Pearson r, …) works directly on them. Functions that genuinely need spatial or temporal extent (SSIM, gradient error, field comparison plots, …) raise a clear error on degenerate inputs; pass `force=True` to bypass with a warning.

---

## Quick start

```python
import numpy as np
import aule
from aule.metrics import rmse, pearson_r, ssim
from aule.plots import plot_field_comparison, plot_scatter

# ── spatial data ────────────────────────────────────────────────────────────
gt   = np.random.rand(8, 64, 64, 1)
pred = gt + np.random.normal(0, 0.1, gt.shape)

print(rmse(gt, pred))
print(pearson_r(gt, pred))
print(ssim(gt, pred))

fig, axes = plot_field_comparison(gt[0], pred[0])
fig, ax   = plot_scatter(gt, pred)

# ── pure time series ─────────────────────────────────────────────────────────
ts_gt   = np.random.randn(4, 3, 200)   # (B=4, C=3, T=200)
ts_pred = np.roll(ts_gt, 5, axis=-1) + 0.05 * np.random.randn(*ts_gt.shape)

print(rmse(ts_gt, ts_pred, axes="bct"))
print(pearson_r(ts_gt, ts_pred, axes="bct"))

from aule.metrics import lag_correlation, dtw_distance
from aule.plots  import plot_lag_correlation, plot_multi_channel_series

corr = lag_correlation(ts_gt, ts_pred, max_lag=30, axes="bct")
dtw  = dtw_distance(ts_gt, ts_pred, axes="bct")

fig, ax  = plot_lag_correlation(ts_gt, ts_pred, max_lag=30, axes="bct")
fig, axs = plot_multi_channel_series(ts_gt, axes="bct",
                                      channel_names=["Temp", "Precip", "Wind"])
```

---

## Logging

aule is silent by default. Enable structured, coloured log output with a single call:

```python
import aule
aule.set_log_level("DEBUG")   # DEBUG | INFO | WARNING | ERROR
# or via environment variable before importing:
# AULE_LOG_LEVEL=INFO python my_script.py
```

---

## Metrics

All importable directly from `aule.metrics`.

**Core** — `rmse`, `mse`, `mae`, `bias`, `pearson_r`, `ssim`, `psnr`, `r2_score`, `mape`, `smape`, `nse`, `kge`, `max_error`, `explained_variance`, `wasserstein_distance`, `quantile_mapping_bias`

**Spectral / gradient** — `spectral_error`, `gradient_error`, `psd_radial_error`, `spectral_angle_mapper`

**Climate** — `seasonal_error`, `percentile_error`, `pixelwise_temporal_correlation`, `trend_error`, `extreme_event_duration_error`, `autocorrelation_error`, `wet_day_frequency_error`, `dry_spell_error`, `anomaly_correlation_coefficient`

**Ensemble** — `ensemble_spread`, `crps`, `rank_histogram`, `brier_score`, `spread_skill_ratio`, `crps_skill_score`

**Earth observation** — `normalized_difference_index`, `index_error`, `change_detection_error`

**Classification / segmentation** — `iou`, `dice`, `precision_recall_f1`, `confusion_matrix_metrics`, `cohen_kappa`

**Uncertainty** — `picp`, `pit_histogram`

**Spatial verification** — `fractions_skill_score` (FSS), `energy_score`

**Time series** — `lag_correlation`, `cross_channel_correlation`, `peak_timing_error`, `dtw_distance`

---

## Plots

All importable directly from `aule.plots`.

**Core** — `plot_scatter`, `plot_qq`, `plot_histogram_comparison`, `plot_error_histogram`

**Spatial** — `plot_field_comparison`, `plot_bias_map`, `plot_correlation_map`

**Climate** — `plot_temporal_trend`, `plot_temporal_scatter`

**Ensemble** — `plot_ensemble_spread_map`, `plot_rank_histogram`

**Diagnostics** — `plot_taylor_diagram`, `plot_boxplot_comparison`, `plot_violin_comparison`, `plot_time_series`, `plot_error_map`

**Classification** — `plot_confusion_matrix`, `plot_reliability_diagram`

**Advanced** — `plot_hovmoller`, `plot_cdf_comparison`, `plot_spectral_density`, `plot_time_evolution`

**Time series** — `plot_lag_correlation`, `plot_multi_channel_series`, `plot_dtw_alignment`, `plot_channel_correlation_matrix`

Spatial plots accept optional `lat`/`lon` arrays for a cartopy basemap (requires `aule[geo]`). Every plot returns `(fig, ax)` and accepts an optional `save_path`.

### Divergent colormap normalizations

`plot_bias_map`, `plot_error_map`, and `plot_field_comparison` accept `norm_type` to control how extreme values stand out:

```python
fig, ax = plot_bias_map(gt, pred, norm_type="symlog", norm_kwargs={"linthresh": 0.02})
# norm_type options: "linear" (default) | "power" | "symlog" | "twoslope"
```

---

## Shape guardrails

Functions that require genuine spatial or temporal extent declare this explicitly. Passing a degenerate input raises a descriptive error:

```python
series_gt = np.random.randn(1, 1, 4)   # H=W=1 — no real spatial extent
ssim(series_gt, series_gt)              # raises ValueError with a clear message

ssim(series_gt, series_gt, force=True)  # proceeds anyway with a warning
```

---

## Object-oriented API

Bind arrays once, call everything as a method — including all new functions:

```python
from aule import aule

v = aule(gt, pred)
print(v.rmse())
print(v.pearson_r())
fig, ax = v.plot_scatter()
fig, ax = v.plot_bias_map(norm_type="power")
```

---

## Automatic validation report

```python
from aule.report import generate_report

generate_report(gt, pred, save_path="report.html")
```

Produces a self-contained HTML file (figures embedded as base64 PNGs) with a metrics table and all key plots.

---

## Documentation

```bash
python build_doc.py
```

Documentation is built with [pdoc](https://pdoc.dev).
