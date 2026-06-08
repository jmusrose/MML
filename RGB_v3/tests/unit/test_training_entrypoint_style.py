#!/usr/bin/env python3
"""Static checks for RGB v3 NYU training entrypoint behavior."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_nyu_entrypoint_defines_early_stopping():
    source = (PROJECT_ROOT / "DML_nyu.py").read_text(encoding="utf-8")

    assert "--patience" in source
    assert "n_no_improve" in source
    assert "if n_no_improve >= args.patience" in source
    assert "Stopping early" in source
    assert "break" in source


def test_nyu_entrypoint_validates_on_test_loader():
    source = (PROJECT_ROOT / "DML_nyu.py").read_text(encoding="utf-8")

    assert "validation_loader = test_loader" in source
    assert "val_acc = val_rgbd(epoch, validation_loader, model, logger, args)" in source
