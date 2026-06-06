"""
DML_CREMAD/dataset/CREMA_noised.py

支持噪声注入的 CramedDataset 变体。
噪声仅施加于视频模态（图像帧），音频保持不变。

支持场景：
- Clean: 无噪声
- Salt & Pepper (Lvl 5.0): density=0.10, p=0.5
- Salt & Pepper (Lvl 10.0): density=0.10, p=1.0
- Gaussian (Lvl 5.0): variance=5, p=0.5
- Gaussian (Lvl 10.0): variance=10, p=0.5
"""

import random
import numpy as np
from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
from PIL import Image, ImageFile
import librosa
import csv

ImageFile.LOAD_TRUNCATED_IMAGES = True

from dataset.noise_transforms import AddSaltPepperNoise, AddGaussianNoise


class CramedDatasetNoised(Dataset):
    """CramedDataset with configurable noise injection on visual modality.

    Parameters
    ----------
    config : dict
        配置字典，包含 dataset.data_root, fps, setting.num_class 等。
    mode : str
        'train' 或 'test'。
    noise_type : str or None
        噪声类型: 'salt_pepper', 'gaussian', 或 None (clean)。
    noise_level : float
        噪声等级: 5.0 或 10.0。
        - salt_pepper: 5.0 -> p=0.5, 10.0 -> p=1.0 (density固定0.10)
        - gaussian: 5.0 -> variance=5, 10.0 -> variance=10 (p固定0.5)
    """

    def __init__(self, config, mode='test', noise_type=None, noise_level=5.0):
        self.config = config
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        self.use_pre_frame = config.get("fps", 3)
        self.data_root = config["dataset"]["data_root"]
        self.noise_type = noise_type
        self.noise_level = noise_level

        class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

        self.train_csv = os.path.join(self.data_root, 'train.csv')
        self.test_csv = os.path.join(self.data_root, 'test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(
                    self.data_root, 'cremad_datasets', 'AudioWAV', item[0] + '.wav'
                )
                visual_path = os.path.join(
                    self.data_root, 'cremad_datasets', 'Image', item[0]
                )
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])

        # Build visual transform with noise
        self.visual_transform = self._build_visual_transform()

    def _build_visual_transform(self):
        """构建视频帧的 transform pipeline，根据噪声配置注入噪声。"""
        transform_list = [
            transforms.Resize(size=(256, 256)),
        ]

        # Add noise transform if specified
        if self.noise_type == 'salt_pepper':
            if self.noise_level >= 10.0:
                p = 1.0
            else:
                p = 0.5
            transform_list.append(
                transforms.RandomApply(
                    [AddSaltPepperNoise(density=0.10, p=1.0)], p=p
                )
            )
        elif self.noise_type == 'gaussian':
            variance = int(self.noise_level) if self.noise_level >= 10.0 else 5
            transform_list.append(
                transforms.RandomApply(
                    [AddGaussianNoise(mean=0.0, variance=variance)], p=0.5
                )
            )

        transform_list.extend([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        # Audio (unchanged - no noise applied)
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 20)[:22050 * 20]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        spectrogram = torch.tensor(spectrogram)

        # Visual (with noise injection)
        image_samples = os.listdir(self.image[idx])
        file_num = len(image_samples)

        if file_num < self.use_pre_frame:
            select_index = random.choices(image_samples, k=self.use_pre_frame)
        else:
            select_index = random.sample(image_samples, self.use_pre_frame)

        select_index.sort()
        images = torch.zeros((self.use_pre_frame, 3, 224, 224))

        for i in range(self.use_pre_frame):
            img = Image.open(
                os.path.join(self.image[idx], select_index[i])
            ).convert('RGB')
            img = self.visual_transform(img)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))

        one_hot = np.eye(self.config["setting"]["num_class"])
        one_hot_label = one_hot[self.label[idx]]
        label = torch.FloatTensor(one_hot_label)

        return spectrogram, images, label
