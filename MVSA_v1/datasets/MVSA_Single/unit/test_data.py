#!/usr/bin/env python3
"""Unit tests for data module: dataset loading, collate function."""

import json
import os
import sys
import tempfile
from argparse import Namespace
from functools import partial

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.dataset import JsonlDataset, AddGaussianNoise, AddSaltPepperNoise
from data.vocab import Vocab
from data.helpers import get_transforms, collate_fn


def _make_tokenizer():
    """Simple whitespace tokenizer for testing without BERT dependency."""
    return lambda text: text.lower().split()


def _make_vocab():
    """Create a simple vocab for testing."""
    vocab = Vocab()
    words = ["i", "love", "this", "movie", "is", "terrible", "it", "okay", "guess"]
    vocab.add(words)
    return vocab


class TestJsonlDataset:
    """Tests for JsonlDataset."""

    def test_dataset_length(self, sample_jsonl_file, mock_args):
        vocab = _make_vocab()
        transforms = get_transforms()
        ds = JsonlDataset(
            sample_jsonl_file, _make_tokenizer(), transforms, "train", vocab, mock_args
        )
        assert len(ds) == 3

    def test_dataset_item_shapes(self, sample_jsonl_file, mock_args):
        vocab = _make_vocab()
        transforms = get_transforms()
        ds = JsonlDataset(
            sample_jsonl_file, _make_tokenizer(), transforms, "train", vocab, mock_args
        )
        sentence, segment, image, label, idx = ds[0]

        assert sentence.dim() == 1
        assert sentence.dtype == torch.long
        assert segment.shape == sentence.shape
        assert image.shape == (3, 224, 224)
        assert label.shape == (1,)
        assert idx.shape == (1,)

    def test_blank_image_when_img_none(self, sample_jsonl_file, mock_args):
        """When img field is None, should produce a valid image tensor."""
        vocab = _make_vocab()
        transforms = get_transforms()
        ds = JsonlDataset(
            sample_jsonl_file, _make_tokenizer(), transforms, "train", vocab, mock_args
        )
        # All samples have img=None in our fixture
        _, _, image, _, _ = ds[0]
        assert image.shape == (3, 224, 224)
        assert not torch.isnan(image).any()

    def test_cls_token_prepended(self, sample_jsonl_file, mock_args):
        vocab = _make_vocab()
        transforms = get_transforms()
        ds = JsonlDataset(
            sample_jsonl_file, _make_tokenizer(), transforms, "train", vocab, mock_args
        )
        sentence, _, _, _, _ = ds[0]
        # First token should be [CLS] index
        assert sentence[0].item() == vocab.stoi["[CLS]"]


class TestCollateFunction:
    """Tests for the collate_fn."""

    def test_collate_single_sample(self, sample_jsonl_file, mock_args):
        vocab = _make_vocab()
        transforms = get_transforms()
        ds = JsonlDataset(
            sample_jsonl_file, _make_tokenizer(), transforms, "train", vocab, mock_args
        )
        batch = [ds[0]]
        text, segment, mask, image, target, indices = collate_fn(batch, mock_args)

        assert text.shape[0] == 1
        assert segment.shape[0] == 1
        assert mask.shape[0] == 1
        assert image.shape[0] == 1
        assert target.shape[0] == 1

    def test_collate_padding(self, sample_jsonl_file, mock_args):
        """Batch with variable-length sequences should be padded."""
        vocab = _make_vocab()
        transforms = get_transforms()
        ds = JsonlDataset(
            sample_jsonl_file, _make_tokenizer(), transforms, "train", vocab, mock_args
        )
        batch = [ds[0], ds[1], ds[2]]
        text, segment, mask, image, target, indices = collate_fn(batch, mock_args)

        bsz = 3
        assert text.shape[0] == bsz
        assert segment.shape[0] == bsz
        assert mask.shape[0] == bsz
        # All padded to same length
        assert text.shape[1] == mask.shape[1]

    def test_collate_mask_sum_equals_original_length(self, sample_jsonl_file, mock_args):
        """Mask sum for each sample should equal its original sequence length."""
        vocab = _make_vocab()
        transforms = get_transforms()
        ds = JsonlDataset(
            sample_jsonl_file, _make_tokenizer(), transforms, "train", vocab, mock_args
        )
        batch = [ds[0], ds[1], ds[2]]
        original_lens = [len(ds[i][0]) for i in range(3)]

        text, segment, mask, image, target, indices = collate_fn(batch, mock_args)

        for i, orig_len in enumerate(original_lens):
            assert mask[i].sum().item() == orig_len


class TestNoiseTransforms:
    """Tests for noise transform classes."""

    def test_gaussian_noise_output_shape(self):
        from PIL import Image

        img = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 128)
        noise = AddGaussianNoise(mean=0.0, variance=1.0, amplitude=5.0)
        result = noise(img)
        assert result.size == (64, 64)

    def test_salt_pepper_noise_output_shape(self):
        from PIL import Image

        img = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 128)
        noise = AddSaltPepperNoise(density=0.1, p=1.0)
        result = noise(img)
        assert result.size == (64, 64)


class TestVocab:
    """Tests for Vocab class."""

    def test_default_vocab_has_special_tokens(self):
        vocab = Vocab()
        assert "[PAD]" in vocab.stoi
        assert "[UNK]" in vocab.stoi
        assert "[CLS]" in vocab.stoi
        assert "[SEP]" in vocab.stoi
        assert "[MASK]" in vocab.stoi

    def test_add_words(self):
        vocab = Vocab()
        initial_sz = vocab.vocab_sz
        vocab.add(["hello", "world"])
        assert vocab.vocab_sz == initial_sz + 2
        assert "hello" in vocab.stoi
        assert "world" in vocab.stoi

    def test_empty_init(self):
        vocab = Vocab(emptyInit=True)
        assert vocab.vocab_sz == 0
        assert len(vocab.stoi) == 0
