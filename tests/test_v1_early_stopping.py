from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CLI_ENTRYPOINTS = [
    ROOT / "RGB_v1" / "DML_nyu.py",
    ROOT / "RGB_v1" / "DML_sun.py",
    ROOT / "CMU_v1" / "DML_mosi.py",
    ROOT / "CMU_v1" / "DML_mosei.py",
    ROOT / "MVSA_v1" / "DML_MVSA.py",
    ROOT / "Food_v1" / "DML_Food.py",
]


def test_v1_cli_entrypoints_expose_uniform_early_stopping_arguments():
    for path in CLI_ENTRYPOINTS:
        source = path.read_text(encoding="utf-8")

        assert "--early_stop_patience" in source, path
        assert "default=3" in source, path
        assert "--early_stop_min_delta" in source, path
        assert "default=0.0" in source, path


def test_v1_cli_entrypoints_stop_after_patience_without_metric_improvement():
    for path in CLI_ENTRYPOINTS:
        source = path.read_text(encoding="utf-8")

        assert "early_stop_counter" in source, path
        assert "args.early_stop_patience" in source, path
        assert "args.early_stop_min_delta" in source, path
        assert "Stopping early" in source, path


def test_cremad_v1_supports_configured_and_cli_overridden_early_stopping():
    source = (ROOT / "CREMAD_v1" / "DML_cremad.py").read_text(encoding="utf-8")
    config = (ROOT / "CREMAD_v1" / "data" / "crema.json").read_text(encoding="utf-8")

    assert "--early_stop_patience" in source
    assert "--early_stop_min_delta" in source
    assert "cfg['train']['early_stop_patience']" in source
    assert "cfg['train']['early_stop_min_delta']" in source
    assert "early_stop_counter" in source
    assert '"early_stop_patience": 3' in config
    assert '"early_stop_min_delta": 0.0' in config
