from types import SimpleNamespace
import importlib
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeDataset:
    def __init__(self, args, data_dir=None, transform=None):
        self.args = args
        self.data_dir = data_dir
        self.transform = transform
        split = Path(data_dir).name
        self.classes = ["class_a", "class_b"]
        count = 10 if split == "train" else 3
        self.imgs = [
            (str(Path(data_dir) / f"class_{idx % 2}" / f"{split}_{idx}.png"), idx % 2)
            for idx in range(count)
        ]

    def __len__(self):
        return len(self.imgs)


class FakeLoader:
    def __init__(self, dataset, batch_size=None, shuffle=None, num_workers=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return self.values


class FakeSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)


def install_training_import_stubs(monkeypatch):
    torch = types.ModuleType("torch")
    torch.randperm = lambda n: FakeTensor(range(n))
    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.DataLoader = object
    torch.utils.data.Subset = FakeSubset
    torch.nn = types.ModuleType("torch.nn")
    torch.optim = types.ModuleType("torch.optim")

    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = types.ModuleType("torchvision.transforms")

    sklearn = types.ModuleType("sklearn")
    sklearn.metrics = types.ModuleType("sklearn.metrics")
    sklearn.metrics.accuracy_score = lambda labels, preds: 0.0
    sklearn.metrics.average_precision_score = lambda labels, preds: 0.0

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **kwargs: iterable

    stubs = {
        "torch": torch,
        "torch.nn": torch.nn,
        "torch.optim": torch.optim,
        "torch.utils": torch.utils,
        "torch.utils.data": torch.utils.data,
        "torchvision": torchvision,
        "torchvision.transforms": torchvision.transforms,
        "sklearn": sklearn,
        "sklearn.metrics": sklearn.metrics,
        "tqdm": tqdm_module,
        "data.additional_transform": types.ModuleType("data.additional_transform"),
        "data.aligned_conc_dataset": types.ModuleType("data.aligned_conc_dataset"),
        "data.aligned_conc_dataset_noised": types.ModuleType("data.aligned_conc_dataset_noised"),
        "models.dml_classifier_nyu": types.ModuleType("models.dml_classifier_nyu"),
        "tool.loss": types.ModuleType("tool.loss"),
        "utils.logger": types.ModuleType("utils.logger"),
        "utils.utils": types.ModuleType("utils.utils"),
    }
    stubs["data.additional_transform"].AddGaussianNoise = object
    stubs["data.additional_transform"].AddSaltPepperNoise = object
    stubs["data.aligned_conc_dataset"].AlignedConcDataset = object
    stubs["data.aligned_conc_dataset_noised"].AlignedConcDataset = object
    stubs["models.dml_classifier_nyu"].Classifier = object
    stubs["tool.loss"].information_bottleneck_classification_loss = object
    stubs["utils.logger"].create_logger = object
    stubs["utils.utils"].Averager = object
    stubs["utils.utils"].append_experiment_record = object
    stubs["utils.utils"].set_seed = object

    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_nyu_dataloaders_use_cpsc_train_split_for_validation(monkeypatch):
    install_training_import_stubs(monkeypatch)
    sys.modules.pop("DML_nyu", None)
    DML_nyu = importlib.import_module("DML_nyu")

    monkeypatch.setattr(DML_nyu, "AlignedConcDataset", FakeDataset)
    monkeypatch.setattr(DML_nyu, "DataLoader", FakeLoader)
    monkeypatch.setattr(DML_nyu, "Subset", FakeSubset, raising=False)

    args = SimpleNamespace(
        data_path="nyud2_trainvaltest",
        batch_sz=4,
        n_workers=0,
        calib_size=0,
    )

    train_loader, val_loader, test_loader, calib_loader = DML_nyu.build_nyu_dataloaders(
        args,
        train_transform="train-transform",
        val_transform="val-transform",
    )

    assert train_loader.dataset.dataset.data_dir.endswith("train")
    assert val_loader.dataset.dataset.data_dir.endswith("train")
    assert test_loader.dataset.data_dir.endswith("test")
    assert calib_loader is val_loader
    assert train_loader.dataset.indices == [4, 5, 6, 7, 8, 9]
    assert val_loader.dataset.indices == [0, 1, 2, 3]
    assert train_loader.dataset.dataset.transform == "train-transform"
    assert val_loader.dataset.dataset.transform == "val-transform"
    assert test_loader.dataset.transform == "val-transform"
    assert train_loader.shuffle is True
    assert val_loader.shuffle is False
    assert test_loader.shuffle is False


def test_infer_n_classes_from_split_datasets(monkeypatch):
    install_training_import_stubs(monkeypatch)
    sys.modules.pop("DML_nyu", None)
    DML_nyu = importlib.import_module("DML_nyu")

    datasets = [
        SimpleNamespace(classes=["a", "b", "c"]),
        SimpleNamespace(classes=["a", "b", "c"]),
    ]

    assert DML_nyu.infer_n_classes_from_datasets(*datasets) == 3


def test_infer_n_classes_from_subset_datasets(monkeypatch):
    install_training_import_stubs(monkeypatch)
    sys.modules.pop("DML_nyu", None)
    DML_nyu = importlib.import_module("DML_nyu")

    base_dataset = SimpleNamespace(classes=["a", "b"])
    subset = FakeSubset(base_dataset, [0])

    assert DML_nyu.infer_n_classes_from_datasets(subset, base_dataset) == 2
