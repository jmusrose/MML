from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RGB_ENTRYPOINTS = [
    ROOT / version / script
    for version in ["RGB_v1", "RGB_v2", "RGB_v3"]
    for script in ["DML_nyu.py", "DML_sun.py"]
]

TEXT_IMAGE_ENTRYPOINTS = [
    ROOT / version / script
    for version in ["MVSA_v1", "MVSA_v2", "MVSA_v3"]
    for script in ["DML_MVSA.py"]
] + [
    ROOT / version / script
    for version in ["Food_v1", "Food_v2", "Food_v3"]
    for script in ["DML_Food.py"]
]


def test_rgb_versions_log_train_branch_losses_and_accs():
    for path in RGB_ENTRYPOINTS:
        source = path.read_text(encoding="utf-8")

        assert "Total Loss" in source
        assert "loss_both:" in source
        assert "loss_rgb:" in source
        assert "loss_depth:" in source
        assert "acc_both:" in source
        assert "acc_rgb:" in source
        assert "acc_depth:" in source


def test_text_image_versions_log_train_branch_losses_and_accs():
    for path in TEXT_IMAGE_ENTRYPOINTS:
        source = path.read_text(encoding="utf-8")

        assert "Total Loss" in source
        assert "loss_fused:" in source
        assert "loss_txt:" in source
        assert "loss_img:" in source
        assert "acc_fused:" in source
        assert "acc_txt:" in source
        assert "acc_img:" in source


def test_ib_versions_keep_logging_train_ib_loss():
    for path in [
        *(ROOT / version / script for version in ["RGB_v2", "RGB_v3"] for script in ["DML_nyu.py", "DML_sun.py"]),
        *(ROOT / version / "DML_MVSA.py" for version in ["MVSA_v2", "MVSA_v3"]),
        *(ROOT / version / "DML_Food.py" for version in ["Food_v2", "Food_v3"]),
    ]:
        source = path.read_text(encoding="utf-8")

        assert "loss_ib:" in source
