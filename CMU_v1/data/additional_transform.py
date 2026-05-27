"""特征空间的额外 transform。

镜像 `DML_v1/RGB_v1/data/additional_transform.py` 的风格，但作用对象是
`(T, D)` 的浮点特征张量（vision / audio 两路），用于 CMU-MOSI / CMU-MOSEI
鲁棒性评估场景。
"""

import random

import torch
from torch import Tensor


class AddFeatureGaussianNoise(object):
    """向 `(T, D)` 浮点特征张量按概率添加独立同分布高斯噪声。

    在 padding 位置（`mask == True`）保持 0，不引入噪声，保证下游
    SequenceEncoder 的 padding mask 语义不被破坏。

    Args:
        mean: 高斯噪声均值。
        std: 高斯噪声标准差。
        p: 触发噪声的概率（独立伯努利试验）。

    Validates: Requirements 6.7
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0, p: float = 1.0):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, feat: Tensor, mask: Tensor | None = None) -> Tensor:
        # feat: (T, D), mask: (T,) bool, True = padding 位置
        if random.random() >= self.p:
            return feat
        noise = torch.randn_like(feat) * self.std + self.mean
        if mask is not None:
            noise = noise.masked_fill(mask.unsqueeze(-1), 0.0)
        return feat + noise
