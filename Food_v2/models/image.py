#!/usr/bin/env python3
"""
Image encoder using ResNet-152 backbone with adaptive pooling.
"""

import torch
import torch.nn as nn
import torchvision


class ImageEncoder(nn.Module):
    """ResNet-152 backbone with adaptive pooling.

    Removes the last FC layer and average pooling from ResNet-152,
    then applies adaptive pooling to produce num_image_embeds patches.

    Output shape: B x num_image_embeds x 2048
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
        """Input: B x 3 x 224 x 224, Output: B x num_image_embeds x 2048"""
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # B x num_image_embeds x 2048


class ImageClf(nn.Module):
    """Image classifier: ImageEncoder + Linear head.

    Flattens the encoder output and maps to n_classes logits.
    """

    def __init__(self, args):
        super(ImageClf, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoder(args)
        self.clf = nn.Linear(
            args.img_hidden_sz * args.num_image_embeds, args.n_classes
        )

    def forward(self, x):
        """Returns (logits [B, n_classes], features [B, 2048*num_embeds])"""
        x = self.img_encoder(x)
        x = torch.flatten(x, start_dim=1)
        out = self.clf(x)
        return out, x
