#!/usr/bin/env python3
"""
Shared test fixtures for DML Food-101 multimodal tests.
"""

import os
import sys
import tempfile
from argparse import Namespace

import numpy as np
import pytest
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_args():
    """Create a mock args namespace with all required parameters for Food-101."""
    args = Namespace(
        batch_sz=4,
        bert_model="bert-base-uncased",
        data_path=os.path.join(os.path.dirname(__file__), "..", "data"),
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
        n_classes=101,
        n_workers=0,
        name="test_food101",
        noise_level=0.0,
        noise_type="Gaussian",
        num_image_embeds=3,
        patience=15,
        savedir=tempfile.mkdtemp(),
        seed=1,
        task="Food101",
    )
    return args


@pytest.fixture
def dummy_dataset_dir(tmp_path):
    """Create a small dummy Food-101 directory structure for integration tests.

    Structure:
        tmp_path/images/train/{class_name}/{id}.jpg
        tmp_path/images/test/{class_name}/{id}.jpg
        tmp_path/texts_txt/{class_name}/{id}.txt

    Creates 3 classes with 2 samples each for train and test.
    """
    classes = ["apple_pie", "baby_back_ribs", "baklava"]

    for split in ["train", "test"]:
        for cls in classes:
            img_dir = tmp_path / "images" / split / cls
            img_dir.mkdir(parents=True, exist_ok=True)

            txt_dir = tmp_path / "texts_txt" / cls
            txt_dir.mkdir(parents=True, exist_ok=True)

            for i in range(2):
                # Create a small dummy image
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(str(img_dir / f"sample_{i}.jpg"))

                # Create a dummy text file
                txt_path = txt_dir / f"sample_{i}.txt"
                txt_path.write_text(
                    f"This is a sample {cls} food item number {i}",
                    encoding="utf-8",
                )

    return str(tmp_path)


@pytest.fixture
def mock_args_with_dummy_data(mock_args, dummy_dataset_dir):
    """Mock args with data_path pointing to the dummy dataset."""
    mock_args.data_path = dummy_dataset_dir
    mock_args.n_classes = 3
    return mock_args


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
