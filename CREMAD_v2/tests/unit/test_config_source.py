#!/usr/bin/env python3
"""CREMAD should read run-shaping parameters from cfg instead of script constants."""

from pathlib import Path
import json

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


def test_cremad_script_allows_information_bottleneck_cli_overrides():
    source = (PROJECT_ROOT / "DML_cremad.py").read_text(encoding="utf-8")

    assert "parser.add_argument('--ib_beta'" in source
    assert "parser.add_argument('--ib_eps_scale'" in source
    assert "cfg['ib_beta'] = args.ib_beta" in source
    assert "cfg['ib_eps_scale'] = args.ib_eps_scale" in source


def test_cremad_v2_config_writes_outputs_under_v2_savepath():
    config = json.loads((PROJECT_ROOT / "data" / "crema.json").read_text(encoding="utf-8"))

    save_dir = config["save_dir"].replace("\\", "/")
    assert "CREMAD_v2/savepath/CREMA-D" in save_dir
    assert "CREMAD_v1/savepath" not in save_dir


def test_cremad_v2_default_lr_matches_cremad_v1_baseline():
    config = json.loads((PROJECT_ROOT / "data" / "crema.json").read_text(encoding="utf-8"))

    assert config["train"]["optimizer"]["lr"] == 0.005
