#!/usr/bin/env python3
"""Unit tests for utils module: Averager, set_seed, checkpoint save/load."""

import os
import sys
import tempfile

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.utils import Averager, set_seed, save_checkpoint, load_checkpoint


class TestAverager:
    """Tests for the Averager class."""

    def test_empty_averager(self):
        avg = Averager()
        assert avg.item() == 0.0
        assert avg.n == 0

    def test_single_value(self):
        avg = Averager()
        avg.add(5.0)
        assert avg.item() == 5.0
        assert avg.n == 1

    def test_multiple_values(self):
        avg = Averager()
        avg.add(2.0)
        avg.add(4.0)
        avg.add(6.0)
        assert abs(avg.item() - 4.0) < 1e-6

    def test_reset(self):
        avg = Averager()
        avg.add(10.0)
        avg.add(20.0)
        avg.reset()
        assert avg.item() == 0.0
        assert avg.n == 0


class TestSetSeed:
    """Tests for seed reproducibility."""

    def test_torch_reproducibility(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_numpy_reproducibility(self):
        set_seed(42)
        a = np.random.randn(5)
        set_seed(42)
        b = np.random.randn(5)
        assert np.allclose(a, b)

    def test_different_seeds_differ(self):
        set_seed(1)
        a = torch.randn(5)
        set_seed(2)
        b = torch.randn(5)
        assert not torch.allclose(a, b)


class TestCheckpoint:
    """Tests for checkpoint save/load round-trip."""

    def test_save_and_load(self, mock_args):
        from models.dml_classifier import Classifier

        # Use small model for testing
        mock_args.n_classes = 3
        model = Classifier(mock_args)

        # Save checkpoint
        state = {
            "epoch": 5,
            "state_dict": model.state_dict(),
            "optimizer": {},
            "scheduler": {},
            "n_no_improve": 0,
            "best_metric": 0.75,
        }

        save_dir = tempfile.mkdtemp()
        save_checkpoint(state, is_best=True, checkpoint_path=save_dir)

        # Load checkpoint
        model2 = Classifier(mock_args)
        load_checkpoint(model2, os.path.join(save_dir, "model_best.pt"))

        # Verify parameters match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    def test_checkpoint_file_created(self):
        save_dir = tempfile.mkdtemp()
        state = {"epoch": 1, "state_dict": {}}
        save_checkpoint(state, is_best=False, checkpoint_path=save_dir)
        assert os.path.exists(os.path.join(save_dir, "checkpoint.pt"))

    def test_best_model_created_when_is_best(self):
        save_dir = tempfile.mkdtemp()
        state = {"epoch": 1, "state_dict": {}}
        save_checkpoint(state, is_best=True, checkpoint_path=save_dir)
        assert os.path.exists(os.path.join(save_dir, "model_best.pt"))
