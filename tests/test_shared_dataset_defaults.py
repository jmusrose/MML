import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_versioned_entrypoints_default_to_shared_datasets():
    expected = {
        "RGB_v1/DML_nyu.py": "nyud2_trainvaltest",
        "RGB_v1/DML_sun.py": "sunrgbd",
        "RGB_v2/DML_nyu.py": "nyud2_trainvaltest",
        "RGB_v2/DML_sun.py": "sunrgbd",
        "Food_v1/DML_Food.py": "Food101",
        "Food_v2/DML_Food.py": "Food101",
        "MVSA_v1/DML_MVSA.py": "MVSA_Single",
        "MVSA_v2/DML_MVSA.py": "MVSA_Single",
        "CMU_v1/DML_mosi.py": "mosi.pkl",
        "CMU_v1/DML_mosei.py": "mosei.pkl",
    }

    for relative_path, dataset_name in expected.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")

        assert "datasets_shared" in text
        assert dataset_name in text


def test_cremad_configs_default_to_shared_dataset_root():
    expected_root = (ROOT / "datasets_shared" / "CREMA-D").as_posix()

    for relative_path in ["CREMAD_v1/data/crema.json", "CREMAD_v2/data/crema.json"]:
        config = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))

        assert config["dataset"]["data_root"] == expected_root
