#!/usr/bin/env python3
"""
Shared test fixtures for DML MVSA tests.
"""

import os
import sys
import json
import tempfile
from argparse import Namespace

import pytest
try:
    import torch
except ModuleNotFoundError:
    torch = None

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_args():
    """Create a mock args namespace with default values for testing."""
    args = Namespace(
        batch_sz=4,
        bert_model="bert-base-uncased",
        data_path="./datasets",
        drop_img_percent=0.0,
        dropout=0.1,
        freeze_img=3,
        freeze_txt=5,
        hidden_sz=768,
        img_embed_pool_type="avg",
        img_hidden_sz=2048,
        lr=3e-5,
        lr_factor=0.5,
        lr_patience=2,
        max_epochs=50,
        max_seq_len=512,
        n_classes=3,
        n_workers=0,
        name="test_run",
        noise_level=0.0,
        noise_type="Gaussian",
        num_image_embeds=3,
        patience=15,
        savedir=tempfile.mkdtemp(),
        seed=1,
        task="MVSA_Single",
        labels=["negative", "neutral", "positive"],
        label_freqs={"negative": 100, "neutral": 200, "positive": 150},
    )
    return args


@pytest.fixture
def sample_jsonl_file(tmp_path):
    """Create a small JSONL fixture file for testing."""
    samples = [
        {"id": "1", "label": "positive", "text": "I love this movie", "img": None},
        {"id": "2", "label": "negative", "text": "This is terrible", "img": None},
        {"id": "3", "label": "neutral", "text": "It is okay I guess", "img": None},
    ]
    jsonl_path = tmp_path / "test.jsonl"
    with open(jsonl_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    return str(jsonl_path)


@pytest.fixture
def device():
    """Return available device."""
    if torch is None:
        pytest.skip("torch is not installed in this environment")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
