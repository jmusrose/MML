#!/usr/bin/env python3
"""Static checks for RGB-style training entrypoint behavior."""

from pathlib import Path


def test_food_entrypoint_uses_rgb_style_result_outputs():
    source = Path("DML_Food.py").read_text(encoding="utf-8")

    assert "append_experiment_record" in source
    assert "all_experiments.json" in source
    assert "model_best_clean.pt" in source
    assert "training.log" in source
    assert "--note" in source


def test_food_entrypoint_uses_readable_training_efficiency_defaults():
    source = Path("DML_Food.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--batch_sz", type=int, default=32' in source
    assert "get_test_loader" in source
    assert "current_test_loader = get_test_loader(args)" in source


def test_food_entrypoint_uses_information_bottleneck_training():
    source = Path("DML_Food.py").read_text(encoding="utf-8")

    assert "--ib_beta" in source
    assert "--ib_eps_scale" in source
    assert "information_bottleneck_classification_loss" in source
    assert '"ib_beta": args.ib_beta' in source
    assert '"ib_eps_scale": args.ib_eps_scale' in source


def test_food_entrypoint_exposes_conformal_uncertainty_tau():
    source = Path("DML_Food.py").read_text(encoding="utf-8")

    assert "--uncertainty_tau" in source
    assert '"tau": float(args.uncertainty_tau)' in source
    assert "tau=args.uncertainty_tau" in source
