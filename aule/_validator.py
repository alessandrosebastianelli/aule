"""
    Object-oriented wrapper around aule's functional API.

    The `aule` class lets you bind `y_true`/`y_pred` (and optionally
    `data_format`/`ignore_nan`) once, then call every metric/plot as a method:

        from aule import aule

        v = aule(y_true, y_pred)
        score = v.rmse()
        fig, ax = v.plot_scatter()

    Every method is generated dynamically from the public functions found in
    `aule.metrics` and `aule.plots` at import time, so adding a new function
    to either subpackage (and exporting it via `__all__`) automatically makes
    it available as a method here too - no manual wiring required. The
    standalone functions (`aule.metrics.rmse`, ...) are untouched and remain
    fully usable on their own.
"""

import inspect
from typing import Optional

from . import metrics as _metrics_module
from . import plots as _plots_module

# Names of the "bound" attributes that are auto-injected into a wrapped
# function call when the function itself declares a parameter with that name.
_BOUND_ATTRS = ("y_true", "y_pred", "data_format", "ignore_nan")


def _collect_public_functions(module) -> dict:
    '''
        Collects every public, function-typed attribute exposed by a module
        (i.e. everything listed in its __all__, or every non-underscore
        callable if __all__ is not defined).

        Parameters:
        -----------
        - module : module
            Module to inspect (e.g. `aule.metrics`, `aule.plots`).

        Returns:
        --------
        - functions : dict
            Mapping from function name to the function object.
    '''

    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]

    functions = {}
    for name in names:
        obj = getattr(module, name, None)
        if inspect.isfunction(obj):
            functions[name] = obj

    return functions


def _make_bound_method(func):
    '''
        Wraps a free function into a method that, when called on an `aule`
        instance, automatically supplies any of `y_true`, `y_pred`,
        `data_format`, `ignore_nan` that both the function accepts and the
        instance has stored - without overriding values explicitly passed
        by the caller.

        Parameters:
        -----------
        - func : callable
            The original function from `aule.metrics` or `aule.plots`.

        Returns:
        --------
        - method : callable
            A function suitable for assignment as a class method.
    '''

    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())
    auto_params = [p for p in _BOUND_ATTRS if p in accepted]

    def method(self, *args, **kwargs):
        for param in auto_params:
            if param not in kwargs:
                value = getattr(self, param, None)
                if value is not None or param in ("ignore_nan",):
                    kwargs.setdefault(param, value)
        return func(*args, **kwargs)

    method.__name__ = func.__name__
    method.__doc__ = func.__doc__
    method.__signature__ = sig
    return method


class aule:
    '''
        Object-oriented entry point exposing every aule metric and plot as
        a bound method, sharing the same ground truth / prediction arrays.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", used automatically by every method that
            accepts this parameter (default: None).
        - ignore_nan : bool
            Used automatically by every method that accepts this parameter
            (default: False).

        Usage:
        ------

        ```python
        import numpy as np
        from aule import aule

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)

        v = aule(gt, pred)
        print(v.rmse())
        print(v.pearson_r())
        fig, ax = v.plot_scatter(save_path="scatter.png")

        # Functions with a non-standard signature (ensembles, extra bands,
        # ...) remain available, but extra required arguments must still be
        # passed explicitly:
        ensemble = pred[np.newaxis] + np.random.normal(0, 0.05, (10, *pred.shape))
        print(v.crps(y_ensemble=ensemble))
        ```
    '''

    def __init__(
        self,
        y_true,
        y_pred,
        data_format: Optional[str] = None,
        ignore_nan: bool = False,
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.data_format = data_format
        self.ignore_nan = ignore_nan


def _attach_methods():
    '''
        Discovers every public function in `aule.metrics` and `aule.plots`
        and attaches it as a bound method on the `aule` class. Called once
        at import time.
    '''

    all_functions = {}
    all_functions.update(_collect_public_functions(_metrics_module))
    all_functions.update(_collect_public_functions(_plots_module))

    for name, func in all_functions.items():
        setattr(aule, name, _make_bound_method(func))


_attach_methods()
