import sys
sys.path += ['.']

import numpy as np
import pytest

from aule._shapes import to_canonical, match_shapes, apply_nan_mask, finite_mask


def test_shape_c_hwc():
    x = np.random.rand(32, 32, 3)
    c = to_canonical(x)
    assert c.shape == (1, 32, 32, 3, 1)


def test_shape_d_hwct():
    x = np.random.rand(32, 32, 3, 5)
    c = to_canonical(x, data_format="hwct")
    assert c.shape == (1, 32, 32, 3, 5)


def test_shape_a_bhwc():
    x = np.random.rand(8, 32, 32, 3)
    c = to_canonical(x, data_format="bhwc")
    assert c.shape == (8, 32, 32, 3, 1)


def test_shape_b_bhwct():
    x = np.random.rand(8, 32, 32, 3, 5)
    c = to_canonical(x)
    assert c.shape == (8, 32, 32, 3, 5)


def test_4d_requires_no_error_with_default_format():
    # default data_format is "bhwc" when ndim == 4 and not specified
    x = np.random.rand(8, 32, 32, 3)
    c = to_canonical(x)
    assert c.shape == (8, 32, 32, 3, 1)


def test_invalid_data_format_raises():
    x = np.random.rand(8, 32, 32, 3)
    with pytest.raises(ValueError):
        to_canonical(x, data_format="invalid")


def test_invalid_ndim_raises():
    x = np.random.rand(32)
    with pytest.raises(ValueError):
        to_canonical(x)


def test_match_shapes_mismatch_raises():
    a = np.random.rand(32, 32, 3)
    b = np.random.rand(16, 16, 3)
    with pytest.raises(ValueError):
        match_shapes(a, b)


def test_apply_nan_mask_replaces_with_median():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, 2.0, np.nan])
    a_clean, b_clean = apply_nan_mask(a, b, ignore_nan=True)
    assert np.all(np.isfinite(a_clean))
    assert np.all(np.isfinite(b_clean))


def test_apply_nan_mask_noop_when_disabled():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, 2.0, np.nan])
    a_out, b_out = apply_nan_mask(a, b, ignore_nan=False)
    assert a_out is a
    assert b_out is b


def test_finite_mask():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, 2.0, np.nan])
    mask = finite_mask(a, b)
    np.testing.assert_array_equal(mask, [True, False, False])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
