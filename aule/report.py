"""
    Automatic validation report generator.

    `generate_report` runs a curated set of metrics and plots on a given
    (y_true, y_pred) pair and assembles them into a single self-contained
    HTML file (figures embedded as base64 PNGs, no external assets), so a
    user can get a comprehensive quick-look validation summary without
    calling each metric/plot function by hand.
"""

import base64
import io
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ._shapes import to_canonical
from . import metrics as _m
from . import plots as _p
from ._logging import logger

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

__all__ = ["generate_report"]


def _fig_to_base64(fig: plt.Figure) -> str:
    '''
        Encodes a matplotlib figure as a base64 PNG string, suitable for
        inline embedding in an HTML <img> tag.

        Parameters:
        -----------
        - fig : matplotlib.figure.Figure
            Figure to encode.

        Returns:
        --------
        - encoded : str
            Base64-encoded PNG data (no data-URI prefix).
    '''

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def _safe_call(fn, *args, **kwargs):
    '''
        Calls a metric/plot function, swallowing exceptions so a single
        inapplicable function (e.g. a climate metric called on data
        without a meaningful time axis) does not abort the whole report.

        Parameters:
        -----------
        - fn : callable
            Function to call.
        - *args, **kwargs
            Arguments forwarded to `fn`.

        Returns:
        --------
        - result : the function's return value, or None if it raised.
        - error : str or None
            The exception message, if any.
    '''

    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        return None, str(exc)


def _format_value(value) -> str:
    '''
        Formats a metric result (float or dict of floats) for display in
        the report table.

        Parameters:
        -----------
        - value : float or dict
            The metric's return value.

        Returns:
        --------
        - formatted : str
            Human-readable string representation.
    '''

    if isinstance(value, dict):
        return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in value.items())
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    title: str = "aule validation report",
) -> str:
    '''
        Generates a self-contained HTML validation report for a (y_true,
        y_pred) pair: a curated table of core/spectral/distributional
        metrics, plus the most informative plots (scatter, histogram
        comparison, field comparison, error map, box plot, Taylor diagram,
        CDF comparison). Metrics/plots that require extra arguments not
        available from `y_true`/`y_pred` alone (e.g. ensemble or
        classification-specific functions) are intentionally left out;
        call them directly for those cases.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - save_path : str
            Path to write the HTML report to.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded wherever the
            underlying metric/plot supports it (default: False).
        - title : str
            Report title, shown at the top of the page.

        Returns:
        --------
        - save_path : str
            The same path passed in, for convenience chaining.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.report import generate_report

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        generate_report(gt, pred, save_path="report.html")
        ```
    '''

    matplotlib.use("Agg")

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)
    has_time_axis = y_true_c.shape[-1] > 1
    has_multi_sample = (y_true_c.shape[0] * y_true_c.shape[-1]) > 1

    # --- metrics table ---
    metric_specs = [
        ("RMSE", _m.rmse), ("MAE", _m.mae), ("MSE", _m.mse), ("Bias", _m.bias),
        ("Pearson r", _m.pearson_r), ("R2 score", _m.r2_score), ("SSIM", _m.ssim),
        ("PSNR", _m.psnr), ("MAPE (%)", _m.mape), ("SMAPE (%)", _m.smape),
        ("KGE", _m.kge), ("Max error", _m.max_error), ("Explained variance", _m.explained_variance),
        ("Spectral error", _m.spectral_error), ("Gradient error", _m.gradient_error),
        ("PSD radial error", _m.psd_radial_error),
        ("Wasserstein distance", _m.wasserstein_distance),
        ("Quantile mapping bias", _m.quantile_mapping_bias),
    ]

    metric_rows = []
    metric_iter = _tqdm(metric_specs, desc="report: metrics") if _HAS_TQDM else metric_specs
    for name, fn in metric_iter:
        logger.debug("report: computing %s", name)
        value, error = _safe_call(fn, y_true, y_pred, data_format=data_format, ignore_nan=ignore_nan)
        metric_rows.append((name, _format_value(value) if error is None else f"n/a ({error})"))

    if has_multi_sample:
        logger.debug("report: computing pixelwise_temporal_correlation")
        value, error = _safe_call(_m.pixelwise_temporal_correlation, y_true, y_pred, data_format=data_format, ignore_nan=ignore_nan)
        if error is None:
            metric_rows.append(("Mean pixelwise temporal correlation", _format_value(float(np.mean(value)))))

    if has_time_axis:
        time_metrics = [
            ("Seasonal error", _m.seasonal_error),
            ("Trend error", _m.trend_error),
            ("Autocorrelation error", _m.autocorrelation_error),
        ]
        for name, fn in time_metrics:
            logger.debug("report: computing %s", name)
            value, error = _safe_call(fn, y_true, y_pred, data_format=data_format, ignore_nan=ignore_nan)
            metric_rows.append((name, _format_value(value) if error is None else f"n/a ({error})"))

    # --- plots ---
    plot_specs = [
        ("Scatter: ground truth vs prediction",
         lambda: _p.plot_scatter(y_true, y_pred, data_format=data_format, ignore_nan=ignore_nan)),
        ("Distribution comparison",
         lambda: _p.plot_histogram_comparison(y_true, y_pred, data_format=data_format, ignore_nan=ignore_nan)),
        ("CDF comparison",
         lambda: _p.plot_cdf_comparison(y_true, y_pred, data_format=data_format, ignore_nan=ignore_nan)),
        ("Error distribution",
         lambda: _p.plot_error_histogram(y_true, y_pred, data_format=data_format, ignore_nan=ignore_nan)),
        ("Field comparison (first slice)",
         lambda: _p.plot_field_comparison(y_true, y_pred, data_format=data_format)),
        ("Absolute error map (first slice)",
         lambda: _p.plot_error_map(y_true, y_pred, data_format=data_format)),
        ("Box plot comparison",
         lambda: _p.plot_boxplot_comparison(y_true, y_pred, data_format=data_format, ignore_nan=ignore_nan)),
        ("Taylor diagram",
         lambda: _p.plot_taylor_diagram(y_true, y_pred, data_format=data_format, ignore_nan=ignore_nan)),
    ]
    if has_time_axis:
        plot_specs.append(
            ("Spatial-mean temporal trend",
             lambda: _p.plot_temporal_trend(y_true, y_pred, data_format=data_format, ignore_nan=ignore_nan))
        )

    plot_images = []
    plot_iter = _tqdm(plot_specs, desc="report: plots") if _HAS_TQDM else plot_specs
    for caption, fn in plot_iter:
        logger.debug("report: generating plot '%s'", caption)
        result, err = _safe_call(fn)
        if result is not None:
            fig, _ = result
            plot_images.append((caption, _fig_to_base64(fig)))
        else:
            logger.warning("report: plot '%s' failed: %s", caption, err)

    # --- assemble HTML ---
    rows_html = "\n".join(
        f"<tr><td>{name}</td><td>{value}</td></tr>" for name, value in metric_rows
    )

    images_html = "\n".join(
        f'<div class="plot-card"><h3>{caption}</h3>'
        f'<img src="data:image/png;base64,{encoded}" alt="{caption}"/></div>'
        for caption, encoded in plot_images
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 40px; color: #222; background: #fafafa; }}
  h1 {{ margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 32px; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 700px; margin-bottom: 40px; background: white; }}
  th, td {{ text-align: left; padding: 8px 14px; border-bottom: 1px solid #e0e0e0; }}
  th {{ background: #f0f0f0; }}
  .plots {{ display: flex; flex-wrap: wrap; gap: 24px; }}
  .plot-card {{ background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .plot-card h3 {{ margin: 0 0 8px 0; font-size: 14px; color: #444; }}
  .plot-card img {{ max-width: 480px; display: block; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">Generated by aule on {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<h2>Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{rows_html}
</table>

<h2>Plots</h2>
<div class="plots">
{images_html}
</div>

</body>
</html>"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)

    return save_path
