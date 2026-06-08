#!/usr/bin/env python3
"""
DML Logit Fusion Classifier for CREMA-D emotion recognition.

Decision-level fusion: 0.5 * audio_logits + 0.5 * video_logits.
Audio and video classifier heads use RGB_v2-style information bottlenecks.
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

    Contains two independent encoders and two independent information bottleneck
    classification heads.
    Fusion: fused_logits = 0.5 * audio_logits + 0.5 * video_logits
    """

    def __init__(self, config):
        super(DMLClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config)
        self.video_encoder = VideoEncoder(config, config['fps'])
        self.hidden_dim = 512
        num_classes = config['setting']['num_class']
        self.ib_eps_scale = config.get("ib_eps_scale", 1.0)

        self.audio_mu = nn.Linear(self.hidden_dim, num_classes)
        self.audio_logvar = nn.Linear(self.hidden_dim, num_classes)
        self.video_mu = nn.Linear(self.hidden_dim, num_classes)
        self.video_logvar = nn.Linear(self.hidden_dim, num_classes)

    def _sample_logits(self, mu, logvar):
        if not self.training or self.ib_eps_scale == 0:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + self.ib_eps_scale * eps * std

    @staticmethod
    def _kl_to_standard_normal(mu, logvar):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=1).mean()

    def forward(self, audio, video):
        """
        Args:
            audio: [B, 1, 257, 1249] spectrogram
            video: [B, 3, fps, 224, 224] multi-frame images

        Returns:
            tuple: (
                fused_logits, audio_logits, video_logits,
                audio_latent, video_latent, ib_losses
            )
        """
        audio_features = self.audio_encoder(audio)
        video_features = self.video_encoder(video)

        audio_mu = self.audio_mu(audio_features)
        audio_logvar = self.audio_logvar(audio_features)
        video_mu = self.video_mu(video_features)
        video_logvar = self.video_logvar(video_features)

        audio_logits = self._sample_logits(audio_mu, audio_logvar)
        video_logits = self._sample_logits(video_mu, video_logvar)
        ib_losses = {
            "audio": self._kl_to_standard_normal(audio_mu, audio_logvar),
            "video": self._kl_to_standard_normal(video_mu, video_logvar),
        }

        fused_logits = 0.5 * audio_logits + 0.5 * video_logits

        return fused_logits, audio_logits, video_logits, audio_logits, video_logits, ib_losses
