"""
    Declarative shape-requirement guardrails (private module, not part of
    the public API).

    Some metrics/plots are only meaningful when their input has genuine
    spatial extent (e.g. `gradient_error`, `ssim`, `plot_field_comparison`)
    or genuine temporal extent (e.g. `trend_error`, `autocorrelation_error`).
    Calling them on a degenerate input (H=W=1 for spatial, T=1 for
    temporal - which is exactly what a pure time series promoted via
    `aule._shapes.to_canonical(..., axes=...)` looks like) would either
    crash confusingly deep inside the implementation, or silently return a
    number that looks valid but means nothing (e.g. a "spatial gradient"
    on a 1x1 image is just zero everywhere).

    The `requires` decorator lets a function declare its real requirements
    once, validated automatically against the array(s) it receives. The
    check is bypassed with `force=True` (an extra keyword the decorator
    injects into the function's signature), which logs a warning and lets
    the call through with whatever degenerate shape it has - the caller
    has explicitly said "I know this isn't really spatial/temporal data,
    proceed anyway".

    This only wraps functions whose primary array arguments are still in
    their *original* (non-canonical) shape, since the decorator itself
    calls `to_canonical` to inspect them - it does not change what the
    wrapped function receives, it only inspects and validates.
"""

import functools
import inspect
import warnings
from typing import Callable, Optional

import numpy as np

from ._shapes import to_canonical, is_degenerate_spatial, is_degenerate_temporal
from ._logging import logger


def requires(
    spatial: bool = False,
    temporal: bool = False,
    array_args: Optional[tuple] = None,
) -> Callable:
    '''
        Decorator declaring that a function needs genuine spatial and/or
        temporal extent in its input array(s) to produce a meaningful
        result. Injects a `force` keyword-only parameter into the wrapped
        function: when `force=False` (default) and the requirement isn't
        met, raises a clear `ValueError`; when `force=True`, logs a
        warning and proceeds anyway, after promoting the input to a
        non-degenerate shape if needed (size-1 padding, the same thing
        `to_canonical` already does for series inputs).

        Parameters:
        -----------
        - spatial : bool
            If True, the decorated function requires H > 1 or W > 1 in at
            least one of the inspected array arguments (default: False).
        - temporal : bool
            If True, the decorated function requires T > 1 in at least
            one of the inspected array arguments (default: False).
        - array_args : tuple of str
            Names of the function's parameters that hold input arrays to
            inspect, in the order they should be checked (default:
            `("y_true",)` if present in the signature, else the first
            positional parameter). Only the first array found is used to
            decide degeneracy, since aule's convention is that y_true and
            y_pred always share the same shape.

        Usage:
        ------

        ```python
        from .._guards import requires

        @requires(spatial=True)
        def gradient_error(y_true, y_pred, data_format=None, ignore_nan=False, norm="l1", force=False):
            ...

        # raises ValueError: gradient_error needs spatial extent
        gradient_error(series_gt, series_pred, axes="bct")

        # proceeds anyway, with a warning
        gradient_error(series_gt, series_pred, axes="bct", force=True)
        ```
    '''

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if array_args is not None:
            inspect_names = array_args
        elif "y_true" in params:
            inspect_names = ("y_true",)
        else:
            inspect_names = (params[0],)

        @functools.wraps(func)
        def wrapper(*args, force: bool = False, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            data_format = bound.arguments.get("data_format")
            axes = bound.arguments.get("axes")

            target_array = None
            for name in inspect_names:
                candidate = bound.arguments.get(name)
                if isinstance(candidate, np.ndarray):
                    target_array = candidate
                    break

            if target_array is None:
                # nothing to inspect (shouldn't normally happen); just call through
                return func(*args, **kwargs)

            canonical = to_canonical(target_array, data_format=data_format, axes=axes)

            problems = []
            if spatial and is_degenerate_spatial(canonical):
                problems.append("spatial extent (H=W=1)")
            if temporal and is_degenerate_temporal(canonical):
                problems.append("temporal extent (T=1)")

            if problems:
                message = (
                    f"{func.__name__} requires genuine {' and '.join(problems)}, "
                    f"but the input is degenerate. This usually means a pure time "
                    f"series (or single-pixel/single-step input) was passed to a "
                    f"function designed for spatial/temporal fields, so the result "
                    f"would not be meaningful."
                )
                if not force:
                    raise ValueError(
                        message + " Pass force=True to proceed anyway (the result "
                        "will likely be degenerate, e.g. a spatial gradient of "
                        "zero everywhere)."
                    )
                warnings.warn(message + " Proceeding because force=True was given.", stacklevel=2)
                logger.warning("%s called with force=True on a degenerate input %s", func.__name__, target_array.shape)

            return func(*args, **kwargs)

        # extend the public signature with the injected `force` parameter,
        # so introspection-based tooling (e.g. aule's OO wrapper) still sees it
        new_params = list(sig.parameters.values()) + [
            inspect.Parameter("force", inspect.Parameter.KEYWORD_ONLY, default=False, annotation=bool)
        ]
        wrapper.__signature__ = sig.replace(parameters=new_params)

        return wrapper

    return decorator
