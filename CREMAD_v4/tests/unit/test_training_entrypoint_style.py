#!/usr/bin/env python3
"""Static checks for RGB-style CREMAD training entrypoint behavior."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_cremad_entrypoint_uses_rgb_style_result_outputs():
    source = (PROJECT_ROOT / "DML_cremad.py").read_text(encoding="utf-8")

    assert "append_experiment_record" in source
    assert "all_experiments.json" in source
    assert "model_best_clean.pt" in source
    assert "training.log" in source
    assert "final_results.json" in source


def test_cremad_entrypoint_exposes_conformal_uncertainty_tau():
    source = (PROJECT_ROOT / "DML_cremad.py").read_text(encoding="utf-8")

    assert "--uncertainty_tau" in source
    assert '"tau": float(cfg["uncertainty_tau"])' in source
    assert "tau=cfg['uncertainty_tau']" in source


def test_cremad_entrypoint_defines_early_stopping():
    source = (PROJECT_ROOT / "DML_cremad.py").read_text(encoding="utf-8")

    assert "--patience" in source
    assert "early_stopping_patience" in source
    assert "n_no_improve" in source
    assert "if n_no_improve >= early_stopping_patience" in source
    assert "Stopping early" in source
    assert "break" in source
