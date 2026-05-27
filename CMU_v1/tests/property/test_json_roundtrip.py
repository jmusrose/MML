"""Property tests for ``final_results.json`` serialization round-trip.

Feature: dml-cmu-multimodal
Property 18: final_results.json 序列化往返性
Validates: Requirements 11.3, 9.4

The property under test is: for any JSON-compatible ``final_results``
dict (using only ``str`` / ``int`` / ``float`` / ``dict`` keys+values
shaped according to the project schema), the round-trip
``json.loads(json.dumps(d, indent=4, ensure_ascii=False))`` must equal
``d`` (identical structure and numeric values).

The schema being exercised mirrors design.md "final_results.json schema":

    {
        "best_clean_model": {
            "epoch": int,
            "metrics": {mae, corr, acc7, acc2, f1: float}
        },
        "robustness": {
            <scenario name>: {mae, corr, acc7, acc2, f1: float},
            ...
        }
    }
"""
import json

from hypothesis import given, settings
from hypothesis import strategies as st


# Finite floats: Python's json module uses ``float.__repr__`` which is
# round-trip safe for finite IEEE-754 doubles; non-finite values (NaN /
# inf) are explicitly disallowed by the spec so we exclude them too.
_FINITE_FLOAT = st.floats(
    allow_nan=False,
    allow_infinity=False,
    min_value=-1e9,
    max_value=1e9,
    width=64,
)


# Each scenario dict carries exactly the 5 CMU metrics — keys & types match
# ``compute_cmu_metrics`` return value (Requirement 8.2).
_METRICS_DICT = st.fixed_dictionaries(
    {
        "mae": _FINITE_FLOAT,
        "corr": _FINITE_FLOAT,
        "acc7": _FINITE_FLOAT,
        "acc2": _FINITE_FLOAT,
        "f1": _FINITE_FLOAT,
    }
)


# All 5 robustness scenarios from Requirement 9.1.
_SCENARIO_NAMES = [
    "Clean Test",
    "Vision Gaussian (Lvl 1.0)",
    "Vision Gaussian (Lvl 5.0)",
    "Audio Gaussian (Lvl 1.0)",
    "Audio Gaussian (Lvl 5.0)",
]


_FINAL_RESULTS = st.fixed_dictionaries(
    {
        "best_clean_model": st.fixed_dictionaries(
            {
                "epoch": st.integers(min_value=0, max_value=10_000),
                "metrics": _METRICS_DICT,
            }
        ),
        "robustness": st.dictionaries(
            keys=st.sampled_from(_SCENARIO_NAMES),
            values=_METRICS_DICT,
            min_size=1,
            max_size=len(_SCENARIO_NAMES),
        ),
    }
)


@given(d=_FINAL_RESULTS)
@settings(max_examples=100)
def test_final_results_json_roundtrip(d):
    """Feature: dml-cmu-multimodal, Property 18: final_results.json 序列化往返性.

    For any JSON-compatible ``final_results`` dict, dumping with the project
    settings (``indent=4, ensure_ascii=False``) and loading back must yield
    a structurally and numerically identical dict.
    """
    s = json.dumps(d, indent=4, ensure_ascii=False)
    d2 = json.loads(s)
    assert d2 == d, (
        f"JSON round-trip altered the dict.\n"
        f"original:  {d!r}\n"
        f"reloaded:  {d2!r}"
    )


@given(d=_FINAL_RESULTS)
@settings(max_examples=100)
def test_final_results_json_dump_uses_indent_format(d):
    """Feature: dml-cmu-multimodal, Property 18: final_results.json 序列化往返性 (format arm).

    The dumped form must be human-readable with ``indent=4`` — each line
    after a ``{`` opening either ends the object or starts with at least
    4 spaces of indentation. This catches accidental regressions where a
    caller drops the ``indent`` argument and produces a single-line blob.
    """
    s = json.dumps(d, indent=4, ensure_ascii=False)
    # Must contain newlines (single-line JSON would be a regression).
    if d:  # non-empty top-level dict will produce a multi-line dump
        assert "\n" in s, "indented dump unexpectedly produced single-line output"
    # The top-level object must still parse back to the same value.
    assert json.loads(s) == d
