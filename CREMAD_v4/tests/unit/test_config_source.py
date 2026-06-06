#!/usr/bin/env python3
"""CREMAD should read run-shaping parameters from cfg instead of script constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_cremad_script_uses_cfg_for_validation_and_runtime_parameters():
    source = (PROJECT_ROOT / "DML_cremad.py").read_text(encoding="utf-8")

    assert "cfg['train']['val_size']" in source
    assert "cfg['train']['val_batch_size']" in source
    assert "cfg['train']['val_num_workers']" in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu_id", 0))' in source
    assert "[:16]" not in source
    assert "batch_size=16" not in source
    assert "num_workers=8" not in source
