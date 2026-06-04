from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cremad_versions_log_train_branch_losses():
    for version in ["CREMAD_v1", "CREMAD_v2", "CREMAD_v3"]:
        source = (ROOT / version / "DML_cremad.py").read_text(encoding="utf-8")

        assert "Average Training Loss" in source
        assert "loss_fused:" in source
        assert "loss_audio:" in source
        assert "loss_video:" in source
        assert "acc_fused:" in source
        assert "acc_audio:" in source
        assert "acc_video:" in source


def test_cremad_ib_versions_log_train_ib_loss():
    for version in ["CREMAD_v2", "CREMAD_v3"]:
        source = (ROOT / version / "DML_cremad.py").read_text(encoding="utf-8")

        assert "loss_ib:" in source
