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


def test_cremad_entrypoint_supports_ib_warmup_before_early_stopping():
    source = (PROJECT_ROOT / "DML_cremad.py").read_text(encoding="utf-8")

    assert "--ib_warmup_epochs" in source
    assert "cfg['ib_warmup_epochs'] = args.ib_warmup_epochs" in source
    assert "effective_ib_beta = cfg.get(\"ib_beta\", 1e-3) if epoch >= cfg.get(\"ib_warmup_epochs\", 0) else 0.0" in source
    assert "kl_enabled = epoch >= cfg.get('ib_warmup_epochs', 0)" in source
    assert "if not kl_enabled:" in source
    assert "continue" in source
