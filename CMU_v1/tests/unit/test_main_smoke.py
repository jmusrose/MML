"""Smoke test for ``DML_mosi.py main()`` end-to-end pipeline.

Feature: dml-cmu-multimodal
Task 8.1: 用合成 mini pkl 跑通 ``DML_mosi.py main()`` 的 1 epoch、1 batch 路径.

The test:
  1. Builds a small synthetic pkl (~10 samples split into train/dev/test)
     matching the on-disk record schema
     ``((words, vision, audio), label_arr, meta)``.
  2. Runs ``DML_mosi.py`` end-to-end as a subprocess for 1 epoch with tiny
     batch / sequence / model dimensions, pointing ``--data_path`` and
     ``--savedir`` at the synthetic pkl and the test ``tmp_path``.
  3. Verifies that the expected output artifacts are written:
     - ``{savedir}/{name}/model_best_clean.pt``
     - ``{savedir}/{name}/training.log``
     - ``{savedir}/{name}/final_results.json`` with the 5 robustness
       scenarios as keys and the 5 CMU metric keys
       (``mae`` / ``corr`` / ``acc7`` / ``acc2`` / ``f1``) under each.

Validates: Requirements 1.2, 9.4, 10.2
"""
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest


CMU_V1_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DML_MOSI_SCRIPT = os.path.join(CMU_V1_ROOT, "DML_mosi.py")

# MOSI feature dimensions (matched to the main()-injected defaults so we do
# not need to override --vision_dim / --audio_dim on the CLI).
VISION_DIM = 47
AUDIO_DIM = 74


def _make_record(rng: np.random.Generator, T: int, label: float, idx: int):
    """Build a single raw record matching the on-disk pkl format used by
    :class:`AlignedConcDataset` (``((words, vision, audio), label_arr, meta)``).
    """
    words = [f"w{idx}_{j}" for j in range(max(1, T // 2))]
    vision = rng.standard_normal((T, VISION_DIM)).astype(np.float32)
    audio = rng.standard_normal((T, AUDIO_DIM)).astype(np.float32)
    label_arr = np.array([[label]], dtype=np.float32)
    meta = f"video_{idx}"
    return ((words, vision, audio), label_arr, meta)


def _make_synthetic_pkl(out_path: str) -> None:
    """Write a ~10-sample MOSI-shaped pkl to ``out_path``.

    Splits cover both the padding (T < max_seq_len) and truncation
    (T > max_seq_len) branches, and span both signs of the regression label
    so Acc-2 / F1 / Pearson correlation have non-degenerate inputs.
    """
    rng = np.random.default_rng(0)
    split_specs = {
        "train": [(5, -2.0), (10, 0.0), (15, 1.5), (3, 2.5)],
        "dev":   [(7, -1.0), (12, 0.5)],
        "test":  [(4, 3.0), (10, -0.5), (8, 1.0), (6, -2.0)],
    }
    data = {
        split: [_make_record(rng, T, lab, i) for i, (T, lab) in enumerate(specs)]
        for split, specs in split_specs.items()
    }
    with open(out_path, "wb") as f:
        pickle.dump(data, f)


@pytest.fixture
def mini_pkl(tmp_path):
    pkl_path = str(tmp_path / "mini.pkl")
    _make_synthetic_pkl(pkl_path)
    return pkl_path


def test_dml_mosi_main_smoke(tmp_path, mini_pkl):
    """End-to-end smoke run of ``DML_mosi.py`` on a synthetic pkl.

    Exercises the full ``main()`` pipeline: data load → train (1 epoch) →
    val on test split → best-checkpoint save → 5-scenario robustness eval
    → ``final_results.json`` write → ``all_experiments.json`` append.
    """
    savedir_root = str(tmp_path / "savepath")
    cmd = [
        sys.executable,
        DML_MOSI_SCRIPT,
        "--data_path", mini_pkl,
        "--savedir", savedir_root,
        "--seed", "0",
        "--lr", "1e-4",
        "--max_epochs", "1",
        "--batch_sz", "2",
        "--n_workers", "0",
        "--max_seq_len", "8",
        "--bert_max_len", "12",
        "--hidden_sz", "16",
        "--num_heads", "2",
        "--num_layers", "1",
        "--conv_kernel_size", "3",
    ]
    proc = subprocess.run(
        cmd,
        cwd=CMU_V1_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"DML_mosi.py exited with code {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )

    # main() rewrites args.name = f"dml_mosi_seed{seed}_lr{lr}" and joins it
    # onto args.savedir; lr=1e-4 stringifies as '0.0001'.
    run_dir = os.path.join(savedir_root, "dml_mosi_seed0_lr0.0001")

    # 1. Best clean checkpoint must exist (Requirement 1.2 / 11.1).
    assert os.path.isfile(os.path.join(run_dir, "model_best_clean.pt")), (
        f"missing model_best_clean.pt under {run_dir}"
    )

    # 2. training.log must exist (Requirement 10.2).
    assert os.path.isfile(os.path.join(run_dir, "training.log")), (
        f"missing training.log under {run_dir}"
    )

    # 3. final_results.json must exist with the expected schema
    # (Requirement 9.4).
    final_path = os.path.join(run_dir, "final_results.json")
    assert os.path.isfile(final_path), f"missing final_results.json under {run_dir}"

    with open(final_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    expected_scenarios = {
        "Clean Test",
        "Vision Gaussian (Lvl 1.0)",
        "Vision Gaussian (Lvl 5.0)",
        "Audio Gaussian (Lvl 1.0)",
        "Audio Gaussian (Lvl 5.0)",
    }
    expected_metric_keys = {"mae", "corr", "acc7", "acc2", "f1"}

    assert "robustness" in results, "final_results.json missing 'robustness' section"
    robustness = results["robustness"]
    assert set(robustness.keys()) == expected_scenarios, (
        f"unexpected robustness scenarios: "
        f"{set(robustness.keys()) ^ expected_scenarios}"
    )
    for name, metrics in robustness.items():
        assert set(metrics.keys()) == expected_metric_keys, (
            f"scenario {name!r}: unexpected metric keys "
            f"{set(metrics.keys()) ^ expected_metric_keys}"
        )
        for k, v in metrics.items():
            assert isinstance(v, float), (
                f"scenario {name!r}, metric {k!r}: expected float, got {type(v)}"
            )

    # best_clean_model section: epoch + 5-metric dict.
    assert "best_clean_model" in results
    bcm = results["best_clean_model"]
    assert isinstance(bcm.get("epoch"), int)
    assert set(bcm.get("metrics", {}).keys()) == expected_metric_keys
