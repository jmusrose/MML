#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CMU-MOSI / CMU-MOSEI 已对齐多模态数据集（Dataset）。

镜像 `DML_v1/RGB_v1/data/aligned_conc_dataset.py` 的命名风格，但底层数据
格式不再是图像目录，而是 `mosi.pkl` / `mosei.pkl` 中已对齐的 (text words,
vision, audio, label) 四元组。

每个样本被预先解析、tokenize 并 pad/truncate 后缓存到内存，`__getitem__`
直接返回字典，避免在训练循环中反复触发 BertTokenizer 与 numpy→tensor 的
转换开销。

样本字典 schema（与 design.md 一致）::

    {
        'vision':              FloatTensor[T_max, V_dim],
        'audio':               FloatTensor[T_max, A_dim],
        'vision_mask':         BoolTensor[T_max],   # True = padding
        'audio_mask':          BoolTensor[T_max],   # True = padding
        'text_input_ids':      LongTensor[L_max],
        'text_attention_mask': LongTensor[L_max],
        'label':               FloatTensor (scalar),
        'idx':                 int,
        'meta':                str | tuple[str, ...],
    }

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.8, 6.9, 13.3, 13.4
"""

import os
import pickle
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _pad_or_truncate(arr: np.ndarray, max_len: int, dim: int) -> Tuple[Tensor, Tensor]:
    """把变长 `(T, dim)` 序列对齐到 `(max_len, dim)`。

    - `T <= max_len`：右侧用 0 padding 至 `max_len`，对应 mask 位置标记 True。
    - `T >  max_len`：截断到前 `max_len` 步，mask 全 False。

    Args:
        arr: 原始序列，形状 `(T, dim)`，dtype 任意可被 ``float()`` 接受。
        max_len: 目标长度。
        dim: 特征维度（用于 padding 的 0 张量）。

    Returns:
        feat: ``FloatTensor[max_len, dim]``。
        mask: ``BoolTensor[max_len]``，True 表示 padding 位置。

    Validates: Requirements 6.3, 14.2
    """
    feat = torch.zeros(max_len, dim, dtype=torch.float32)
    mask = torch.ones(max_len, dtype=torch.bool)  # True = padding
    T = int(arr.shape[0])
    keep = min(T, max_len)
    if keep > 0:
        feat[:keep] = torch.from_numpy(np.asarray(arr[:keep], dtype=np.float32))
        mask[:keep] = False
    return feat, mask


def _normalize_meta(meta):
    """把每条记录的 meta 字段规范成 collate 友好的简单 Python 类型。

    - MOSI 中 meta 是 ``str`` —— 原样返回。
    - MOSEI 中 meta 是 ``np.ndarray`` of strings —— 转成 ``tuple[str, ...]``。
    - list / tuple —— 转成 ``tuple[str, ...]``。
    """
    if isinstance(meta, str):
        return meta
    if isinstance(meta, np.ndarray):
        return tuple(str(x) for x in meta.tolist())
    if isinstance(meta, (list, tuple)):
        return tuple(str(x) for x in meta)
    return str(meta)


class AlignedConcDataset(Dataset):
    """CMU-MOSI / CMU-MOSEI 已对齐多模态数据集。

    Args:
        cfg: argparse Namespace；至少需要包含
            ``vision_dim, audio_dim, max_seq_len, bert_max_len`` 四个字段
            （以及可选的 ``dataset`` 字段，仅用于打印日志）。
        pkl_path: 数据 pkl 路径，必须存在。
        split: ``"train" | "dev" | "test"``。
        tokenizer: 已实例化的 HuggingFace tokenizer
            （例如 ``BertTokenizer.from_pretrained(args.bert_model_name)``）。

    Raises:
        FileNotFoundError: 当 ``pkl_path`` 指向的文件不存在。
        ValueError: 当对应 split 在 pkl 中为空列表。
    """

    def __init__(self, cfg, pkl_path: str, split: str, tokenizer):
        self.cfg = cfg
        self.pkl_path = pkl_path
        self.split = split
        self.tokenizer = tokenizer

        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(
                f"AlignedConcDataset: pkl file not found: {pkl_path} (split={split})"
            )

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if split not in data:
            raise ValueError(
                f"AlignedConcDataset: split '{split}' not found in {pkl_path}; "
                f"available splits = {list(data.keys())}"
            )

        records = data[split]
        if len(records) == 0:
            raise ValueError(
                f"AlignedConcDataset: split '{split}' is empty in {pkl_path}"
            )

        max_seq_len = int(cfg.max_seq_len)
        vision_dim = int(cfg.vision_dim)
        audio_dim = int(cfg.audio_dim)
        bert_max_len = int(cfg.bert_max_len)

        self.samples = []
        for idx, record in enumerate(records):
            (words, vision_arr, audio_arr), label_arr, meta = record

            # text → BERT tokens
            text = " ".join(str(w) for w in words)
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=bert_max_len,
                return_tensors="pt",
            )
            text_input_ids = enc["input_ids"].squeeze(0).long()
            text_attention_mask = enc["attention_mask"].squeeze(0).long()

            # vision / audio → fixed-length padded sequences
            vision_feat, vision_mask = _pad_or_truncate(
                np.asarray(vision_arr), max_seq_len, vision_dim
            )
            audio_feat, audio_mask = _pad_or_truncate(
                np.asarray(audio_arr), max_seq_len, audio_dim
            )

            # label → float scalar tensor
            label_val = float(np.asarray(label_arr).flatten()[0])
            label = torch.tensor(label_val, dtype=torch.float32)

            self.samples.append({
                "vision": vision_feat,
                "audio": audio_feat,
                "vision_mask": vision_mask,
                "audio_mask": audio_mask,
                "text_input_ids": text_input_ids,
                "text_attention_mask": text_attention_mask,
                "label": label,
                "idx": idx,
                "meta": _normalize_meta(meta),
            })

        dataset_name = getattr(cfg, "dataset", "cmu")
        print(f"[{dataset_name}/{split}] num_samples = {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]
