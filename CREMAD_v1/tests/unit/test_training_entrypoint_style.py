#!/usr/bin/env python3
"""Static checks for RGB-style CREMAD training entrypoint behavior."""

from pathlib import Path


def test_cremad_entrypoint_uses_rgb_style_result_outputs():
    source = Path("DML_cremad.py").read_text(encoding="utf-8")

    assert "append_experiment_record" in source
    assert "all_experiments.json" in source
    assert "model_best_clean.pt" in source
    assert "training.log" in source
    assert "final_results.json" in source
