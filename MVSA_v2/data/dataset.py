#!/usr/bin/env python3
"""
Dataset and noise transforms for DML MVSA multimodal sentiment analysis.
"""

import json
import os
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from data.vocab import Vocab


class JsonlDataset(Dataset):
    """Dataset for multimodal JSONL files supporting text, image and label data."""

    def __init__(self, data_path, tokenizer, transforms, mode, vocab, args):
        """
        Args:
            data_path: Path to .jsonl file
            tokenizer: BERT tokenizer.tokenize function
            transforms: torchvision image transforms
            mode: 'train' or 'test' (controls noise application)
            vocab: Vocabulary object with stoi/itos mappings
            args: Configuration namespace
        """
        self.data = [json.loads(l) for l in open(data_path, encoding="utf-8")]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"]
        self.mode = mode

        # Randomly drop images during training based on drop rate
        np.random.seed(0)
        for row in self.data:
            if np.random.random() < args.drop_img_percent:
                row["img"] = None

        self.max_seq_len = args.max_seq_len
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Returns (sentence_ids, segment_ids, image_tensor, label, index)"""
        tokens = self.tokenizer(self.data[index]["text"])

        # Apply word replacement noise in test mode
        if self.args.noise_level > 0.0 and self.mode == "test":
            if np.random.choice([0, 1], p=[0.5, 0.5]):
                wordlist = self.data[index]["text"].split(" ")
                for i in range(len(wordlist)):
                    replace_p = 0.1 * self.args.noise_level
                    replace_flag = np.random.choice(
                        [0, 1], p=[1 - replace_p, replace_p]
                    )
                    if replace_flag:
                        wordlist[i] = "_"
                noised_text = " ".join(wordlist)
                tokens = self.tokenizer(noised_text)

        # Add CLS token and truncate
        sentence = self.text_start_token + tokens[: (self.max_seq_len - 1)]
        segment = torch.zeros(len(sentence))

        # Convert tokens to vocabulary indices
        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        # Label
        label = torch.LongTensor([self.args.labels.index(self.data[index]["label"])])

        # Process image
        if self.data[index].get("img"):
            img_path = os.path.join(self.data_dir, self.data[index]["img"])
            try:
                image = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, OSError):
                image = Image.fromarray(
                    128 * np.ones((256, 256, 3), dtype=np.uint8)
                )
        else:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        image = self.transforms(image)

        return sentence, segment, image, label, torch.LongTensor([index])


class AddGaussianNoise(object):
    """Add Gaussian noise to PIL images."""

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(
            loc=self.mean, scale=self.variance, size=(h, w, 1)
        )
        N = np.repeat(N, c, axis=2)
        img = N + img
        img = np.clip(img, 0, 255)
        return Image.fromarray(img.astype("uint8")).convert("RGB")


class AddSaltPepperNoise(object):
    """Add salt-and-pepper noise to PIL images."""

    def __init__(self, density=0, p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd

            mask = np.random.choice(
                (0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd]
            )
            mask = np.repeat(mask, c, axis=2)

            img[mask == 0] = 0
            img[mask == 1] = 255

            img = Image.fromarray(img.astype("uint8")).convert("RGB")
        return img
