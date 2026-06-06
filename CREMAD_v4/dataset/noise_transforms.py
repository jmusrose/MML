"""
DML_CREMAD/dataset/noise_transforms.py

视频模态噪声注入 transforms，参考 DML_v1/RGB_v1/data/additional_transform.py。

支持：
- AddSaltPepperNoise: 椒盐噪声 (density=0.10, p=0.5/1.0)
- AddGaussianNoise: 高斯噪声 (variance=5/10, p=0.5)

噪声仅施加于视频模态（PIL Image），音频保持不变。
"""

import numpy as np
import random
from PIL import Image


class AddSaltPepperNoise(object):
    """椒盐噪声 transform，作用于 PIL Image。

    Parameters
    ----------
    density : float
        噪声密度，控制被替换像素的比例。
    p : float
        应用噪声的概率。
    """

    def __init__(self, density=0.10, p=0.5):
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
            img[mask == 0] = 0    # pepper
            img[mask == 1] = 255  # salt
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img


class AddGaussianNoise(object):
    """高斯噪声 transform，作用于 PIL Image。

    Parameters
    ----------
    mean : float
        噪声均值。
    variance : float
        噪声标准差（scale）。
    amplitude : float
        噪声幅度缩放因子。
    """

    def __init__(self, mean=0.0, variance=1.0, amplitude=1):
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
        img[img > 255] = 255
        img[img < 0] = 0
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
