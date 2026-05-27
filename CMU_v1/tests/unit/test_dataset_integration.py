"""Integration test for ``AlignedConcDataset`` and the noised variant.

Feature: dml-cmu-multimodal
Task 4.1: 用合成 mini pkl 验证 ``AlignedConcDataset`` 与
``AlignedConcDataset_noised`` 全流程可正确返回字典样本。

The test constructs a small in-memory pkl (with ``train``/``dev``/``test``
splits) following the on-disk record schema documented in the design
(``((words, vision, audio), label_arr, meta)``) and exercises the public
dataset APIs end-to-end. It validates:

- Schema of the returned sample dict (keys, dtypes, shapes).
- Pad / truncate semantics on both vision and audio modalities, including
  the ``True = padding`` mask convention.
- Label parsing (scalar float tensor, ∈ [-3, 3]).
- Length of each split matches the synthetic data.
- Modality isolation in ``AlignedConcDataset_noised``: when only one of the
  two transforms is provided, the other modality and all non-modality
  fields stay byte-identical to the clean dataset.

Validates: Requirements 6.1, 6.2, 6.6
"""
import argparse
import os
import pickle

import numpy as np
import pytest
import torch

from data.additional_transform import AddFeatureGaussianNoise
from data.cmu_aligned_dataset import AlignedConcDataset
from data.cmu_aligned_dataset_noised import AlignedConcDataset as AlignedConcDatasetNoised


VISION_DIM = 47
AUDIO_DIM = 74
MAX_SEQ_LEN = 10
BERT_MAX_LEN = 12


def _build_cfg() -> argparse.Namespace:
    return argparse.Namespace(
        dataset="mosi_synth",
        vision_dim=VISION_DIM,
        audio_dim=AUDIO_DIM,
        max_seq_len=MAX_SEQ_LEN,
        bert_max_len=BERT_MAX_LEN,
    )


def _make_record(rng: np.random.Generator, T: int, label: float, idx: int):
    """Build a single raw record matching the on-disk pkl format."""
    words = [f"w{idx}_{j}" for j in range(max(1, T // 2))]
    vision = rng.standard_normal((T, VISION_DIM)).astype(np.float32)
    audio = rng.standard_normal((T, AUDIO_DIM)).astype(np.float32)
    label_arr = np.array([[label]], dtype=np.float32)
    meta = f"video_{idx}"
    return ((words, vision, audio), label_arr, meta)


def _make_synthetic_pkl(tmp_path) -> str:
    rng = np.random.default_rng(0)
    # Mix of T < max_seq_len, T == max_seq_len, T > max_seq_len so we cover
    # both the padding and the truncation branch of ``_pad_or_truncate``.
    split_specs = {
        "train": [(5, -2.0), (10, 0.0), (15, 1.5), (3, 2.5)],
        "dev":   [(7, -1.0), (12, 0.5)],
        "test":  [(4, 3.0), (10, -0.5), (8, 0.0)],
    }
    data = {
        split: [_make_record(rng, T, lab, i) for i, (T, lab) in enumerate(specs)]
        for split, specs in split_specs.items()
    }
    pkl_path = os.path.join(str(tmp_path), "mini.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    return pkl_path


@pytest.fixture(scope="module")
def tokenizer():
    """Load a real BERT tokenizer once per module (cached locally)."""
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def mini_pkl(tmp_path):
    return _make_synthetic_pkl(tmp_path)


# --------------------------------------------------------------------------- #
# AlignedConcDataset                                                          #
# --------------------------------------------------------------------------- #


def _assert_sample_schema(sample: dict, expected_label: float):
    """Per-sample schema check (keys, dtypes, shapes, value ranges)."""
    expected_keys = {
        "vision", "audio", "vision_mask", "audio_mask",
        "text_input_ids", "text_attention_mask",
        "label", "idx", "meta",
    }
    assert set(sample.keys()) == expected_keys, (
        f"unexpected sample keys: {set(sample.keys()) ^ expected_keys}"
    )

    assert sample["vision"].shape == (MAX_SEQ_LEN, VISION_DIM)
    assert sample["vision"].dtype == torch.float32
    assert sample["audio"].shape == (MAX_SEQ_LEN, AUDIO_DIM)
    assert sample["audio"].dtype == torch.float32

    assert sample["vision_mask"].shape == (MAX_SEQ_LEN,)
    assert sample["vision_mask"].dtype == torch.bool
    assert sample["audio_mask"].shape == (MAX_SEQ_LEN,)
    assert sample["audio_mask"].dtype == torch.bool

    assert sample["text_input_ids"].shape == (BERT_MAX_LEN,)
    assert sample["text_input_ids"].dtype == torch.long
    assert sample["text_attention_mask"].shape == (BERT_MAX_LEN,)
    assert sample["text_attention_mask"].dtype == torch.long

    assert sample["label"].dtype == torch.float32
    assert sample["label"].ndim == 0
    assert float(sample["label"].item()) == pytest.approx(expected_label)

    assert isinstance(sample["idx"], int)
    assert isinstance(sample["meta"], str)


def test_aligned_dataset_returns_correct_schema_per_split(mini_pkl, tokenizer):
    cfg = _build_cfg()
    expected_lengths = {"train": 4, "dev": 2, "test": 3}
    expected_labels = {
        "train": [-2.0, 0.0, 1.5, 2.5],
        "dev":   [-1.0, 0.5],
        "test":  [3.0, -0.5, 0.0],
    }

    for split, exp_len in expected_lengths.items():
        ds = AlignedConcDataset(cfg, mini_pkl, split, tokenizer)
        assert len(ds) == exp_len, f"split={split}: len mismatch"

        for i, label in enumerate(expected_labels[split]):
            sample = ds[i]
            _assert_sample_schema(sample, expected_label=label)


def test_aligned_dataset_padding_semantics(mini_pkl, tokenizer):
    """T < max_seq_len: trailing positions are zeros and mask=True."""
    cfg = _build_cfg()
    ds = AlignedConcDataset(cfg, mini_pkl, "train", tokenizer)

    # Sample 0 was constructed with T=5 < MAX_SEQ_LEN=10.
    sample = ds[0]
    valid_T = 5

    # Valid positions: mask=False; padded positions: mask=True.
    assert torch.all(~sample["vision_mask"][:valid_T])
    assert torch.all(sample["vision_mask"][valid_T:])
    assert torch.all(~sample["audio_mask"][:valid_T])
    assert torch.all(sample["audio_mask"][valid_T:])

    # Padded positions must be exactly zero.
    assert torch.all(sample["vision"][valid_T:] == 0.0)
    assert torch.all(sample["audio"][valid_T:] == 0.0)


def test_aligned_dataset_truncation_semantics(mini_pkl, tokenizer):
    """T > max_seq_len: mask is all False, length capped at max_seq_len."""
    cfg = _build_cfg()
    ds = AlignedConcDataset(cfg, mini_pkl, "train", tokenizer)

    # Sample 2 was constructed with T=15 > MAX_SEQ_LEN=10.
    sample = ds[2]
    assert sample["vision"].shape[0] == MAX_SEQ_LEN
    assert sample["audio"].shape[0] == MAX_SEQ_LEN
    assert torch.all(~sample["vision_mask"])
    assert torch.all(~sample["audio_mask"])


def test_aligned_dataset_missing_pkl_raises(tmp_path, tokenizer):
    cfg = _build_cfg()
    missing = os.path.join(str(tmp_path), "does_not_exist.pkl")
    with pytest.raises(FileNotFoundError):
        AlignedConcDataset(cfg, missing, "train", tokenizer)


def test_aligned_dataset_empty_split_raises(tmp_path, tokenizer):
    cfg = _build_cfg()
    pkl_path = os.path.join(str(tmp_path), "empty.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"train": [], "dev": [], "test": []}, f)

    with pytest.raises(ValueError):
        AlignedConcDataset(cfg, pkl_path, "train", tokenizer)


# --------------------------------------------------------------------------- #
# AlignedConcDataset_noised                                                   #
# --------------------------------------------------------------------------- #


def test_noised_dataset_no_transforms_matches_clean(mini_pkl, tokenizer):
    """With both transforms=None, noised dataset is byte-identical to clean."""
    cfg = _build_cfg()
    clean = AlignedConcDataset(cfg, mini_pkl, "test", tokenizer)
    noised = AlignedConcDatasetNoised(
        cfg, mini_pkl, "test", tokenizer,
        vision_transform=None, audio_transform=None,
    )

    assert len(clean) == len(noised)
    for i in range(len(clean)):
        s_c, s_n = clean[i], noised[i]
        for key in ("vision", "audio", "vision_mask", "audio_mask",
                    "text_input_ids", "text_attention_mask", "label"):
            assert torch.equal(s_c[key], s_n[key]), (
                f"sample {i}: field {key!r} drifted with no transforms"
            )
        assert s_c["idx"] == s_n["idx"]
        assert s_c["meta"] == s_n["meta"]


def test_noised_dataset_modality_isolation(mini_pkl, tokenizer):
    """Vision noise must not perturb audio / text / mask / label fields."""
    cfg = _build_cfg()
    clean = AlignedConcDataset(cfg, mini_pkl, "test", tokenizer)
    vision_noise = AddFeatureGaussianNoise(mean=0.0, std=1.0, p=1.0)
    noised = AlignedConcDatasetNoised(
        cfg, mini_pkl, "test", tokenizer,
        vision_transform=vision_noise, audio_transform=None,
    )

    torch.manual_seed(0)
    for i in range(len(clean)):
        s_c, s_n = clean[i], noised[i]

        # Audio / text / masks / label / meta must be untouched.
        for key in ("audio", "vision_mask", "audio_mask",
                    "text_input_ids", "text_attention_mask", "label"):
            assert torch.equal(s_c[key], s_n[key]), (
                f"sample {i}: field {key!r} should be unchanged by vision noise"
            )

        # Vision should differ at non-padding positions, but stay 0 at
        # padding positions (the transform respects the mask).
        pad_positions = s_c["vision_mask"]
        if pad_positions.any():
            assert torch.all(s_n["vision"][pad_positions] == 0.0), (
                f"sample {i}: vision padding positions must remain 0"
            )


def test_noised_dataset_audio_isolation(mini_pkl, tokenizer):
    """Symmetric check: audio noise must not perturb vision."""
    cfg = _build_cfg()
    clean = AlignedConcDataset(cfg, mini_pkl, "test", tokenizer)
    audio_noise = AddFeatureGaussianNoise(mean=0.0, std=1.0, p=1.0)
    noised = AlignedConcDatasetNoised(
        cfg, mini_pkl, "test", tokenizer,
        vision_transform=None, audio_transform=audio_noise,
    )

    torch.manual_seed(0)
    for i in range(len(clean)):
        s_c, s_n = clean[i], noised[i]
        assert torch.equal(s_c["vision"], s_n["vision"]), (
            f"sample {i}: vision should be unchanged by audio noise"
        )

        pad_positions = s_c["audio_mask"]
        if pad_positions.any():
            assert torch.all(s_n["audio"][pad_positions] == 0.0), (
                f"sample {i}: audio padding positions must remain 0"
            )
