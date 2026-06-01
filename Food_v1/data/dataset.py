#!/usr/bin/env python3
"""
Dataset and noise transforms for DML Food-101 multimodal classification.
"""

import os
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from data.vocab import Vocab


class Food101Dataset(Dataset):
    """Dataset for UPMC Food-101 multimodal classification.

    Loads image-text pairs organized by directory structure:
        data_root/images/{split}/{class_name}/{id}.jpg
        data_root/texts_clean/{class_name}/{id}.txt
        data_root/texts_txt/{class_name}/{id}.txt

    Uses cleaned text first, falls back to raw text, and handles
    missing/corrupt image files with a gray placeholder.
    """

    def __init__(self, data_root, split, tokenizer, transforms, mode, vocab, args):
        """
        Args:
            data_root: Root data directory containing images/ and text folders
            split: 'train' or 'test'
            tokenizer: BERT tokenizer.tokenize function
            transforms: torchvision image transforms
            mode: 'train' or 'test' (controls noise application)
            vocab: Vocabulary object with stoi/itos mappings
            args: Configuration namespace
        """
        self.data_root = data_root
        self.split = split
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.mode = mode
        self.vocab = vocab
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.text_start_token = ["[CLS]"]

        # Build sample list by scanning image directory
        self.samples = []
        img_split_dir = os.path.join(data_root, "images", split)
        clean_texts_dir = os.path.join(data_root, "texts_clean")
        raw_texts_dir = os.path.join(data_root, "texts_txt")

        # Get sorted class names for consistent label indexing
        self.classes = sorted(
            [d for d in os.listdir(img_split_dir)
             if os.path.isdir(os.path.join(img_split_dir, d))]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_img_dir = os.path.join(img_split_dir, cls_name)
            label = self.class_to_idx[cls_name]

            for img_file in os.listdir(cls_img_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(cls_img_dir, img_file)
                # Derive text paths: prefer cleaned text and fall back to raw text.
                base_name = os.path.splitext(img_file)[0]
                clean_text_path = os.path.join(
                    clean_texts_dir, cls_name, base_name + ".txt"
                )
                raw_text_path = os.path.join(
                    raw_texts_dir, cls_name, base_name + ".txt"
                )
                self.samples.append((img_path, clean_text_path, raw_text_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """Returns (token_ids, segment_ids, image_tensor, label, index)"""
        img_path, clean_text_path, raw_text_path, label = self.samples[index]

        # Load text
        text = self._load_text(clean_text_path) or self._load_text(raw_text_path)

        # Apply text noise in test mode
        if self.args.noise_level > 0.0 and self.mode == "test":
            text = self._apply_text_noise(text)

        # Tokenize
        tokens = self.tokenizer(text)
        sentence = self.text_start_token + tokens[: (self.max_seq_len - 1)]
        segment = torch.zeros(len(sentence))

        # Convert tokens to vocabulary indices
        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        # Load and transform image
        image = self._load_image(img_path)
        image = self.transforms(image)

        label_tensor = torch.LongTensor([label])
        return sentence, segment, image, label_tensor, torch.LongTensor([index])

    def _load_text(self, text_path):
        """Load text file, return empty string if missing or unreadable."""
        if not os.path.exists(text_path):
            return ""
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except (IOError, UnicodeDecodeError):
            return ""

    def _load_image(self, img_path):
        """Load image, return gray placeholder if missing or corrupt."""
        try:
            image = Image.open(img_path).convert("RGB")
            return image
        except (FileNotFoundError, OSError):
            return Image.fromarray(
                128 * np.ones((256, 256, 3), dtype=np.uint8)
            )

    def _apply_text_noise(self, text):
        """Apply random word replacement noise to text."""
        if np.random.choice([0, 1], p=[0.5, 0.5]):
            wordlist = text.split(" ")
            for i in range(len(wordlist)):
                replace_p = 0.1 * self.args.noise_level
                replace_flag = np.random.choice(
                    [0, 1], p=[1 - replace_p, replace_p]
                )
                if replace_flag:
                    wordlist[i] = "_"
            text = " ".join(wordlist)
        return text


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
