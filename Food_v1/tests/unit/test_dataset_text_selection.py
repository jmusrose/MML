#!/usr/bin/env python3
"""Tests for Food-101 text source selection."""

from argparse import Namespace
from pathlib import Path

from data.dataset import Food101Dataset


def test_dataset_prefers_clean_text_and_falls_back_to_raw_text(tmp_path):
    data_root = tmp_path
    img_dir = data_root / "images" / "train" / "apple_pie"
    raw_dir = data_root / "texts_txt" / "apple_pie"
    clean_dir = data_root / "texts_clean" / "apple_pie"
    img_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    clean_dir.mkdir(parents=True)

    (img_dir / "clean_sample.jpg").write_text("not used")
    (img_dir / "fallback_sample.jpg").write_text("not used")
    (raw_dir / "clean_sample.txt").write_text("raw text")
    (raw_dir / "fallback_sample.txt").write_text("fallback raw text")
    (clean_dir / "clean_sample.txt").write_text("clean text")
    (clean_dir / "fallback_sample.txt").write_text("")

    dataset = Food101Dataset(
        str(data_root),
        "train",
        tokenizer=str.split,
        transforms=lambda image: image,
        mode="train",
        vocab=None,
        args=Namespace(max_seq_len=16, noise_level=0.0),
    )

    text_by_name = {
        Path(path).stem: dataset._load_text(clean_path) or dataset._load_text(raw_path)
        for path, clean_path, raw_path, _ in dataset.samples
    }

    assert text_by_name["clean_sample"] == "clean text"
    assert text_by_name["fallback_sample"] == "fallback raw text"
