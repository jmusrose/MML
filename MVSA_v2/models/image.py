#!/usr/bin/env python3
"""
Image encoder using ResNet-152 backbone with adaptive pooling.
"""

import torch
import torch.nn as nn
import torchvision


class ImageEncoder(nn.Module):
    """ResNet-152 backbone with adaptive pooling.

    Output shape: B x num_image_embeds x img_hidden_sz (2048)
    """

    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        """Input: B x 3 x 224 x 224, Output: B x N x 2048"""
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # B x N x 2048


class ImageClf(nn.Module):
    """Image classifier with RGB_v2-style information bottleneck head."""

    def __init__(self, args):
        super(ImageClf, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoder(args)
        self.ib_eps_scale = getattr(args, "ib_eps_scale", 1.0)
        self.mu = nn.Linear(
            args.img_hidden_sz * args.num_image_embeds, args.n_classes
        )
        self.logvar = nn.Linear(
            args.img_hidden_sz * args.num_image_embeds, args.n_classes
        )

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

    def forward(self, x):
        """Returns (sampled_logits, flattened_features, kl_loss)."""
        x = self.img_encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        out = self._sample_logits(mu, logvar)
        ib_loss = self._kl_to_standard_normal(mu, logvar)
        return out, x, ib_loss
