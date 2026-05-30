#!/usr/bin/env python3
"""Static checks for RGB-style training entrypoint behavior."""

from pathlib import Path


def test_mvsa_entrypoint_uses_rgb_style_result_outputs():
    source = Path("DML_MVSA.py").read_text(encoding="utf-8")

    assert "append_experiment_record" in source
    assert "all_experiments.json" in source
    assert "model_best_clean.pt" in source
    assert "training.log" in source
    assert "--note" in source
