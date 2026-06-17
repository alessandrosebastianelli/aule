import sys
sys.path += ['.']

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from aule import aule as aule_validator
from aule import metrics as metrics_module
from aule import plots as plots_module


def _public_function_names(module):
    return [n for n in dir(module) if not n.startswith("_") and callable(getattr(module, n))]


def test_class_exposes_every_metric_function():
    v = aule_validator(np.zeros((4, 4, 1)), np.zeros((4, 4, 1)))
    for name in _public_function_names(metrics_module):
        assert hasattr(v, name), f"missing method {name}"


def test_class_exposes_every_plot_function():
    v = aule_validator(np.zeros((4, 4, 1)), np.zeros((4, 4, 1)))
    for name in _public_function_names(plots_module):
        assert hasattr(v, name), f"missing method {name}"


def test_rmse_matches_function_call():
    gt = np.random.rand(8, 16, 16, 1)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    v = aule_validator(gt, pred)
    assert v.rmse() == pytest.approx(metrics_module.rmse(gt, pred), abs=1e-10)


def test_data_format_is_auto_bound():
    gt = np.random.rand(16, 16, 1, 5)
    pred = gt.copy()
    v = aule_validator(gt, pred, data_format="hwct")
    assert v.rmse() == pytest.approx(0.0, abs=1e-10)


def test_explicit_kwarg_overrides_bound_attribute():
    gt = np.random.rand(8, 16, 16, 1)
    pred = gt.copy()
    other_gt = np.random.rand(16, 16, 1, 5)
    v = aule_validator(gt, pred, data_format="bhwc")
    # explicitly pass a different array/format and confirm it is honored
    result = v.rmse(y_true=other_gt, y_pred=other_gt, data_format="hwct")
    assert result == pytest.approx(0.0, abs=1e-10)


def test_plot_method_returns_fig_ax():
    gt = np.random.rand(8, 16, 16, 1)
    pred = gt + np.random.normal(0, 0.05, gt.shape)
    v = aule_validator(gt, pred)
    fig, ax = v.plot_scatter()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_non_standard_signature_method_still_callable():
    gt = np.random.rand(16, 16, 1)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 16, 16, 1))
    v = aule_validator(gt, pred)
    score = v.crps(y_ensemble=ensemble)
    assert isinstance(score, float)


def test_standalone_functions_still_work_independently():
    gt = np.random.rand(8, 16, 16, 1)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    # standalone function usage must remain untouched
    assert metrics_module.rmse(gt, pred) >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
