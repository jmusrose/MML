import os
import sys
import types
from argparse import Namespace


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def install_data_helper_import_stubs(monkeypatch):
    torch = types.ModuleType("torch")
    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.DataLoader = object
    torch.utils.data.Subset = object

    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = types.ModuleType("torchvision.transforms")

    transformers = types.ModuleType("transformers")
    transformers.BertTokenizer = object

    dataset_module = types.ModuleType("data.dataset")
    dataset_module.JsonlDataset = object
    dataset_module.AddGaussianNoise = object
    dataset_module.AddSaltPepperNoise = object

    vocab_module = types.ModuleType("data.vocab")
    vocab_module.Vocab = object

    for name, module in {
        "torch": torch,
        "torch.utils": torch.utils,
        "torch.utils.data": torch.utils.data,
        "torchvision": torchvision,
        "torchvision.transforms": torchvision.transforms,
        "transformers": transformers,
        "data.dataset": dataset_module,
        "data.vocab": vocab_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def import_helpers(monkeypatch):
    install_data_helper_import_stubs(monkeypatch)
    sys.modules.pop("data.helpers", None)
    from data import helpers

    return helpers


def test_existing_cp_split_is_used_for_calibration(monkeypatch):
    helpers = import_helpers(monkeypatch)
    from data.helpers import resolve_calibration_jsonl

    task_dir = "MVSA_Single"
    cp_path = os.path.join(task_dir, "cp.jsonl")
    monkeypatch.setattr(helpers.os.path, "isfile", lambda path: path == cp_path)

    path, source = resolve_calibration_jsonl(task_dir)

    assert path == cp_path
    assert source == "cp"


def test_dev_split_is_used_when_cp_is_missing(monkeypatch):
    helpers = import_helpers(monkeypatch)
    from data.helpers import resolve_calibration_jsonl

    task_dir = "MVSA_Single"
    dev_path = os.path.join(task_dir, "dev.jsonl")
    monkeypatch.setattr(helpers.os.path, "isfile", lambda path: path == dev_path)

    path, source = resolve_calibration_jsonl(task_dir)

    assert path == dev_path
    assert source == "dev"


def test_train_split_calibration_size_can_be_configured(monkeypatch):
    import_helpers(monkeypatch)
    from data.helpers import resolve_train_split_calib_size

    assert resolve_train_split_calib_size(num_samples=100, requested_size=7) == 7
    assert resolve_train_split_calib_size(num_samples=100, requested_size=0) == 20
    assert resolve_train_split_calib_size(num_samples=5, requested_size=99) == 4


def test_get_data_loaders_accepts_calib_size_in_args(mock_args):
    assert hasattr(mock_args, "calib_size") is False
    args = Namespace(**vars(mock_args), calib_size=3)
    assert args.calib_size == 3
