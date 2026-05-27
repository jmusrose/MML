"""Property tests for ``utils.utils.Averager``.

Feature: dml-cmu-multimodal
Property 15: Averager 累计正确性
Validates: Requirements 7.5
"""
import math

from hypothesis import given, settings
from hypothesis import strategies as st

from utils.utils import Averager


# Bound the float magnitudes to avoid catastrophic cancellation noise from
# truly extreme values; the property is a numerical equivalence statement, so
# a finite, well-behaved domain is the right place to test it.
_FINITE_FLOAT = st.floats(
    allow_nan=False,
    allow_infinity=False,
    min_value=-1e6,
    max_value=1e6,
    width=64,
)


@given(values=st.lists(_FINITE_FLOAT, min_size=1, max_size=200))
@settings(max_examples=100)
def test_averager_running_mean_matches_sum_over_count(values):
    """Feature: dml-cmu-multimodal, Property 15: Averager 累计正确性.

    For any finite, non-empty float sequence ``[v_1, ..., v_n]``, sequentially
    calling ``Averager.add(v_i)`` must leave ``Averager.item()`` numerically
    equal to ``sum(v_i) / n`` within floating-point tolerance.
    """
    avg = Averager()
    for v in values:
        avg.add(v)

    expected = sum(float(v) for v in values) / len(values)
    actual = avg.item()

    # ``Averager`` accumulates as a running float sum, so order-of-operations
    # may differ from ``sum(...) / n``; allow ULP-scale relative tolerance.
    tol = 1e-9 * max(1.0, abs(expected))
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=tol), (
        f"Averager.item()={actual!r} differs from sum/n={expected!r} "
        f"(values={values!r})"
    )


@given(values=st.lists(_FINITE_FLOAT, min_size=1, max_size=50))
@settings(max_examples=100)
def test_averager_reset_returns_to_zero(values):
    """Feature: dml-cmu-multimodal, Property 15: Averager 累计正确性 (reset arm).

    After ``reset()``, the accumulator must report ``0.0`` regardless of any
    history; this is the identity element of the running mean and is required
    by the spec contract used by ``train_cmu`` between epochs.
    """
    avg = Averager()
    for v in values:
        avg.add(v)
    avg.reset()
    assert avg.item() == 0.0
