from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rgb_v2_entrypoints_expose_early_stopping_arguments():
    for relative in ["RGB_v2/DML_nyu.py", "RGB_v2/DML_sun.py"]:
        source = (ROOT / relative).read_text(encoding="utf-8")

        assert "--early_stop_patience" in source
        assert "default=3" in source
        assert "--early_stop_min_delta" in source
        assert "default=0.0" in source


def test_rgb_v2_entrypoints_stop_after_patience_without_test_improvement():
    for relative in ["RGB_v2/DML_nyu.py", "RGB_v2/DML_sun.py"]:
        source = (ROOT / relative).read_text(encoding="utf-8")

        assert "clean_acc = val_rgbd(epoch, test_loader, model, logger, args)" in source
        assert "early_stop_counter" in source
        assert "args.early_stop_patience" in source
        assert "args.early_stop_min_delta" in source
        assert "Stopping early" in source
        assert "break" in source


def test_existing_mvsa_food_v2_entrypoints_have_early_stopping():
    for relative in ["MVSA_v2/DML_MVSA.py", "Food_v2/DML_Food.py"]:
        source = (ROOT / relative).read_text(encoding="utf-8")

        assert "--patience" in source
        assert "n_no_improve" in source
        assert "Stopping early" in source
        assert "break" in source


def test_cremad_v2_supports_configured_early_stopping():
    source = (ROOT / "CREMAD_v2" / "DML_cremad.py").read_text(encoding="utf-8")
    config = (ROOT / "CREMAD_v2" / "data" / "crema.json").read_text(encoding="utf-8")

    assert "cfg['train']['early_stop_patience']" in source
    assert "cfg['train']['early_stop_min_delta']" in source
    assert "early_stop_counter" in source
    assert "Stopping early" in source
    assert '"early_stop_patience": 3' in config
    assert '"early_stop_min_delta": 0.0' in config
