#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CMU-MOSI / CMU-MOSEI 噪声版数据集（独立 vision / audio transform）。

镜像 `DML_v1/RGB_v1/data/aligned_conc_dataset_noised.py` 的设计风格：
与 :class:`data.cmu_aligned_dataset.AlignedConcDataset` 共享解析逻辑（直接
继承复用 ``__init__`` 与 ``_pad_or_truncate``），但额外接受
``vision_transform`` / ``audio_transform`` 两个独立 transform 参数，分别
作用于 ``sample['vision']`` 与 ``sample['audio']``，text 字段不做任何
处理。

每次 ``__getitem__`` 调用都会在缓存样本字典的浅拷贝上施加 transform，
保证：

- 缓存的原始样本永远不被修改（多 epoch 之间数据保持不变）；
- transform 内部基于随机性的噪声每次重新采样（鲁棒性评估常见用法）；
- mask 字段不被 transform 修改（padding 位置由 transform 自行屏蔽）。

Validates: Requirements 6.6, 9.2
"""

from .cmu_aligned_dataset import AlignedConcDataset as _BaseAlignedConcDataset


class AlignedConcDataset(_BaseAlignedConcDataset):
    """CMU 多模态噪声版数据集。

    Args:
        cfg: argparse Namespace；要求与基类一致
            （``vision_dim, audio_dim, max_seq_len, bert_max_len`` 等）。
        pkl_path: 数据 pkl 路径，必须存在。
        split: ``"train" | "dev" | "test"``。
        tokenizer: 已实例化的 HuggingFace tokenizer。
        vision_transform: 可选；``callable(feat: Tensor[T, V_dim],
            mask: BoolTensor[T]) -> Tensor[T, V_dim]``。``None`` 时不施加
            任何 vision 噪声。
        audio_transform: 可选;同上,作用于 audio 模态。

    Raises:
        FileNotFoundError: 当 ``pkl_path`` 指向的文件不存在。
        ValueError: 当对应 split 在 pkl 中为空列表。
    """

    def __init__(
        self,
        cfg,
        pkl_path: str,
        split: str,
        tokenizer,
        vision_transform=None,
        audio_transform=None,
    ):
        super().__init__(cfg, pkl_path, split, tokenizer)
        self.vision_transform = vision_transform
        self.audio_transform = audio_transform

    def __getitem__(self, index: int) -> dict:
        base = self.samples[index]
        # 浅拷贝以避免 transform 替换张量引用时污染缓存
        sample = dict(base)

        if self.vision_transform is not None:
            sample["vision"] = self.vision_transform(
                sample["vision"], sample["vision_mask"]
            )
        if self.audio_transform is not None:
            sample["audio"] = self.audio_transform(
                sample["audio"], sample["audio_mask"]
            )

        return sample
