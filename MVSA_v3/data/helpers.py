#!/usr/bin/env python3
"""
Data loading helpers: transforms, vocabulary, collate, and data loaders.
"""

import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Subset

from data.dataset import JsonlDataset, AddGaussianNoise, AddSaltPepperNoise
from data.vocab import Vocab


def get_transforms():
    """Standard eval transforms: Resize(256), CenterCrop(224), ToTensor, Normalize."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_gaussian_noise_transforms(severity):
    """Transforms with Gaussian noise at given severity."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomApply(
                [AddGaussianNoise(amplitude=severity * 10)], p=0.5
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_salt_noise_transforms(severity):
    """Transforms with Salt & Pepper noise at given severity."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomApply(
                [AddSaltPepperNoise(density=0.1, p=severity / 10)], p=0.5
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_labels_and_frequencies(path):
    """Extract label set and frequencies from JSONL file."""
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path, encoding="utf-8")]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)
    return list(label_freqs.keys()), label_freqs


def get_vocab(args):
    """Build vocabulary from BERT tokenizer."""
    vocab = Vocab()
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)
    return vocab


def collate_fn(batch, args):
    """Custom collate: pads text to max length in batch, stacks images.

    Returns: (text, segment, mask, image, target, indices)
    """
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = torch.stack([row[2] for row in batch])

    tgt_tensor = torch.cat([row[3] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    idx = torch.cat([row[4] for row in batch]).long()
    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, idx


def resolve_calibration_jsonl(task_dir):
    """Return an existing calibration/validation JSONL path and source label."""
    for filename, source in (
        ("cp.jsonl", "cp"),
        ("val.jsonl", "val"),
        ("dev.jsonl", "dev"),
    ):
        path = os.path.join(task_dir, filename)
        if os.path.isfile(path):
            return path, source
    return None, "train_split"


def resolve_validation_jsonl(task_dir):
    """Return validation JSONL if available, otherwise fall back to test JSONL."""
    for filename in ("val.jsonl", "dev.jsonl", "test.jsonl"):
        path = os.path.join(task_dir, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No validation/test JSONL found in {task_dir}")


def resolve_train_split_calib_size(num_samples, requested_size):
    if num_samples < 2:
        raise ValueError("train split needs at least 2 samples for train/calibration")
    if requested_size > 0:
        return min(int(requested_size), num_samples - 1)
    return min(max(1, int(round(num_samples * 0.2))), num_samples - 1)


def limit_calibration_dataset(dataset, requested_size):
    requested_size = int(requested_size)
    if requested_size <= 0:
        return dataset
    return Subset(dataset, list(range(min(requested_size, len(dataset)))))


def get_data_loaders(args):
    """Returns (train_loader, val_loader, cp_loader, test_loaders dict)."""
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    ).tokenize

    train_transforms = get_transforms()

    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, "train.jsonl")
    )
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    task_dir = os.path.join(args.data_path, args.task)
    train_path = os.path.join(task_dir, "train.jsonl")
    val_path = resolve_validation_jsonl(task_dir)
    calibration_path, calibration_source = resolve_calibration_jsonl(task_dir)
    requested_calib_size = int(getattr(args, "calib_size", 0))

    train_dataset = JsonlDataset(
        train_path,
        tokenizer,
        train_transforms,
        "train",
        vocab,
        args,
    )

    dev = JsonlDataset(
        val_path,
        tokenizer,
        train_transforms,
        "train",
        vocab,
        args,
    )

    train_loader_dataset = train_dataset
    if calibration_path is not None:
        cp_dataset = JsonlDataset(
            calibration_path,
            tokenizer,
            train_transforms,
            "train",
            vocab,
            args,
        )
        cp_loader_dataset = limit_calibration_dataset(
            cp_dataset,
            requested_calib_size,
        )
    else:
        calib_size = resolve_train_split_calib_size(
            len(train_dataset),
            requested_calib_size,
        )
        indices = torch.randperm(len(train_dataset)).tolist()
        calib_indices = indices[:calib_size]
        train_indices = indices[calib_size:]
        train_loader_dataset = Subset(train_dataset, train_indices)
        cp_loader_dataset = Subset(train_dataset, calib_indices)
    args.calibration_source = calibration_source

    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(
        train_loader_dataset,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    cp_loader = DataLoader(
        cp_loader_dataset,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        dev,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    # Test loader with optional noise
    if args.noise_level > 0.0:
        if args.noise_type == "Gaussian":
            test_transforms = get_gaussian_noise_transforms(args.noise_level)
        elif args.noise_type == "Salt":
            test_transforms = get_salt_noise_transforms(args.noise_level)
        else:
            test_transforms = train_transforms
    else:
        test_transforms = train_transforms

    test_set = JsonlDataset(
        os.path.join(args.data_path, args.task, "test.jsonl"),
        tokenizer,
        test_transforms,
        "test",
        vocab,
        args,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    test_loaders = {"test": test_loader}

    return train_loader, val_loader, cp_loader, test_loaders
