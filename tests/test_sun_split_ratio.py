from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUN_ENTRYPOINTS = [
    ROOT / "RGB_v1" / "DML_sun.py",
    ROOT / "RGB_v2" / "DML_sun.py",
    ROOT / "RGB_v3" / "DML_sun.py",
    ROOT / "RGB_v4" / "DML_sun.py",
]


def test_all_sun_versions_expose_manual_val_split_ratio():
    for path in SUN_ENTRYPOINTS:
        source = path.read_text(encoding="utf-8")

        assert "--val_split_ratio" in source
        assert "args.val_split_ratio" in source
        assert "0 <" in source and "< 1" in source


def test_sun_v1_v2_use_ratio_split_and_select_best_on_test_loader():
    for relative in ["RGB_v1/DML_sun.py", "RGB_v2/DML_sun.py"]:
        source = (ROOT / relative).read_text(encoding="utf-8")

        assert "val_size = _resolve_train_split_size(num_samples, args.val_split_ratio)" in source
        assert "val_indices = indices[:val_size]" in source
        assert "train_indices = indices[val_size:]" in source
        assert "clean_acc = val_rgbd(epoch, test_loader, model, logger, args)" in source
        assert "val_indices = torch.randperm(num_samples)[:4]" not in source


def test_sun_v3_v4_calibration_split_can_be_set_by_ratio():
    for relative in ["RGB_v3/DML_sun.py", "RGB_v4/DML_sun.py"]:
        source = (ROOT / relative).read_text(encoding="utf-8")

        assert "_resolve_train_split_size(num_samples, args.val_split_ratio)" in source
        assert "calib_size = _resolve_train_split_size" in source
        assert "calib_size" in source
        assert "--calib_size" not in source
        assert "requested_size" not in source


def test_sun_v3_v4_select_best_on_test_loader():
    for relative in ["RGB_v3/DML_sun.py", "RGB_v4/DML_sun.py"]:
        source = (ROOT / relative).read_text(encoding="utf-8")

        assert "validation_loader = test_loader" in source or "val_acc = val_rgbd(epoch, test_loader, model, logger, args)" in source
