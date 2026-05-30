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
from torch.utils.data import DataLoader

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

    train_dataset = JsonlDataset(
        os.path.join(args.data_path, args.task, "train.jsonl"),
        tokenizer,
        train_transforms,
        "train",
        vocab,
        args,
    )

    cp_dataset = JsonlDataset(
        os.path.join(args.data_path, args.task, "cp.jsonl"),
        tokenizer,
        train_transforms,
        "train",
        vocab,
        args,
    )

    dev = JsonlDataset(
        os.path.join(args.data_path, args.task, "test.jsonl"),
        tokenizer,
        train_transforms,
        "train",
        vocab,
        args,
    )

    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    cp_loader = DataLoader(
        cp_dataset,
        batch_size=16,
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
