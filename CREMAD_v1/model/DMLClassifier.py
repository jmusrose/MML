#!/usr/bin/env python3
"""
DML Logit Fusion Classifier for CREMA-D emotion recognition.

Decision-level fusion: 0.5 * audio_logits + 0.5 * video_logits.
No cross-modal interaction layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Resnet import resnet18


class AudioEncoder(nn.Module):
    """Audio encoder using custom ResNet18 (modality='audio').

    Input: audio spectrogram [B, 1, 257, 1249]
    Output: audio feature [B, 512]
    """

    def __init__(self, config=None):
        super(AudioEncoder, self).__init__()
        if config['text']['name'] == 'resnet18':
            self.audio_net = resnet18(modality='audio')

    def forward(self, audio):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        return a


class VideoEncoder(nn.Module):
    """Video encoder using custom ResNet18 (modality='visual').

    Input: video frames [B, 3, fps, 224, 224]
    Output: video feature [B, 512]
    """

    def __init__(self, config=None, fps=3):
        super(VideoEncoder, self).__init__()
        if config['visual']['name'] == 'resnet18':
            self.video_net = resnet18(modality='visual')
        self.fps = fps

    def forward(self, video):
        v = self.video_net(video)
        (_, C, H, W) = v.size()
        B = int(v.size()[0] / self.fps)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        return v


class DMLClassifier(nn.Module):
    """Decision-level fusion classifier for CREMA-D.

    Contains two independent encoders and two independent linear classification heads.
    Fusion: fused_logits = 0.5 * audio_logits + 0.5 * video_logits
    """

    def __init__(self, config):
        super(DMLClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config)
        self.video_encoder = VideoEncoder(config, config['fps'])
        self.hidden_dim = 512
        num_classes = config['setting']['num_class']

        self.cls_a = nn.Linear(self.hidden_dim, num_classes)
        self.cls_v = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, audio, video):
        """
        Args:
            audio: [B, 1, 257, 1249] spectrogram
            video: [B, 3, fps, 224, 224] multi-frame images

        Returns:
            tuple: (fused_logits, audio_logits, video_logits, audio_features, video_features)
        """
        audio_features = self.audio_encoder(audio)
        video_features = self.video_encoder(video)

        audio_logits = self.cls_a(audio_features)
        video_logits = self.cls_v(video_features)

        fused_logits = 0.5 * audio_logits + 0.5 * video_logits

        return fused_logits, audio_logits, video_logits, audio_features, video_features
