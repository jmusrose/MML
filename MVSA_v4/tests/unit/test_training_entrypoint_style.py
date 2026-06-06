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
    assert "--ib_beta" in source
    assert "--ib_eps_scale" in source
    assert "information_bottleneck_classification_loss" in source


def test_mvsa_entrypoint_exposes_conformal_uncertainty_tau():
    source = Path("DML_MVSA.py").read_text(encoding="utf-8")

    assert "--uncertainty_tau" in source
    assert '"tau": float(args.uncertainty_tau)' in source
    assert "tau=args.uncertainty_tau" in source
