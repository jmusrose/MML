#!/usr/bin/env python3
"""
DML Multimodal Classifier for Food-101 classification.

Decision-level fusion: 0.5 * text_logits + 0.5 * image_logits.
Text and image heads include information bottlenecks.
"""

import torch.nn as nn

from models.bert import BertClf
from models.image import ImageClf


class Classifier(nn.Module):
    """Decision-level fusion multimodal classifier.

    Contains:
    - self.txtclf: BertClf instance (text branch)
    - self.imgclf: ImageClf instance (image branch)

    Fusion: fused_logits = 0.5 * txt_logits + 0.5 * img_logits
    """

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.txtclf = BertClf(args)
        self.imgclf = ImageClf(args)

    def forward(self, txt, mask, segment, img):
        """
        Args:
            txt: token IDs [B, seq_len]
            mask: attention mask [B, seq_len]
            segment: segment IDs [B, seq_len]
            img: image tensor [B, 3, 224, 224]

        Returns:
            fused_logits: 0.5 * txt_logits + 0.5 * img_logits [B, n_classes]
            txt_logits: text branch output [B, n_classes]
            img_logits: image branch output [B, n_classes]
            txt_latent: sampled text logits [B, n_classes]
            img_latent: sampled image logits [B, n_classes]
            ib_loss: KL(text) + KL(image)
        """
        txt_logits, _, txt_ib_loss = self.txtclf(txt, mask, segment)
        img_logits, _, img_ib_loss = self.imgclf(img)

        fused_logits = 0.5 * txt_logits + 0.5 * img_logits
        ib_loss = txt_ib_loss + img_ib_loss

        return fused_logits, txt_logits, img_logits, txt_logits, img_logits, ib_loss
