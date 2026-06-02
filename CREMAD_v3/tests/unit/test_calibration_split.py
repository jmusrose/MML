import importlib
import os
import sys
import types


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def install_entrypoint_import_stubs(monkeypatch):
    torch = types.ModuleType("torch")
    torch.nn = types.ModuleType("torch.nn")
    torch.optim = types.ModuleType("torch.optim")
    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.DataLoader = object
    torch.utils.data.Subset = object
    torch.nn.functional = types.ModuleType("torch.nn.functional")

    sklearn = types.ModuleType("sklearn")
    sklearn.metrics = types.ModuleType("sklearn.metrics")
    sklearn.metrics.f1_score = lambda *args, **kwargs: 0.0
    sklearn.metrics.average_precision_score = lambda *args, **kwargs: 0.0

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **kwargs: iterable

    for name, module in {
        "torch": torch,
        "torch.nn": torch.nn,
        "torch.optim": torch.optim,
        "torch.utils": torch.utils,
        "torch.utils.data": torch.utils.data,
        "torch.nn.functional": torch.nn.functional,
        "sklearn": sklearn,
        "sklearn.metrics": sklearn.metrics,
        "tqdm": tqdm_module,
        "data.template": types.ModuleType("data.template"),
        "dataset.CREMA": types.ModuleType("dataset.CREMA"),
        "dataset.CREMA_noised": types.ModuleType("dataset.CREMA_noised"),
        "model.DMLClassifier": types.ModuleType("model.DMLClassifier"),
        "utils.utils": types.ModuleType("utils.utils"),
        "utils.loss": types.ModuleType("utils.loss"),
        "utils.tools": types.ModuleType("utils.tools"),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules["data.template"].config = {}
    sys.modules["dataset.CREMA"].CramedDataset = object
    sys.modules["dataset.CREMA_noised"].CramedDatasetNoised = object
    sys.modules["model.DMLClassifier"].DMLClassifier = object
    sys.modules["utils.utils"].create_logger = object
    sys.modules["utils.utils"].Averager = object
    sys.modules["utils.utils"].append_experiment_record = object
    sys.modules["utils.utils"].deep_update_dict = lambda a, b: b
    sys.modules["utils.loss"].information_bottleneck_classification_loss = object
    sys.modules["utils.tools"].weight_init = object
    sys.modules["utils.tools"].compute_mAP = object
    sys.modules["utils.tools"].setup_seed = object


def test_train_split_calibration_size_can_be_configured(monkeypatch):
    install_entrypoint_import_stubs(monkeypatch)
    sys.modules.pop("DML_cremad", None)
    DML_cremad = importlib.import_module("DML_cremad")

    assert DML_cremad.resolve_train_split_calib_size(100, 7) == 7
    assert DML_cremad.resolve_train_split_calib_size(100, 0) == 20
    assert DML_cremad.resolve_train_split_calib_size(5, 99) == 4
