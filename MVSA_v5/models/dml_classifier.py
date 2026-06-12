#!/usr/bin/env python3
"""
DML Multimodal Classifier for MVSA sentiment analysis.

Decision-level fusion: 0.5 * text_logits + 0.5 * image_logits.
Text and image classifier heads use RGB_v2-style information bottlenecks.
No cross-modal interaction layers.
"""

import torch
import torch.nn as nn

from models.bert import BertClf
from models.image import ImageClf


class Classifier(nn.Module):
    """Main multimodal model following the DML pattern.

    Contains:
    - self.txtclf: BertClf instance (text branch)
    - self.imgclf: ImageClf instance (image branch)
    """

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.txtclf = BertClf(args)
        self.imgclf = ImageClf(args)

    def forward(self, txt, mask, segment, img):
        """
        Returns:
        - fused_logits: 0.5 * txt_logits + 0.5 * img_logits
        - txt_logits: text branch output (B x n_classes)
        - img_logits: image branch output (B x n_classes)
        - txt_latent: sampled text features
        - img_latent: sampled image features
        - ib_loss: KL(text) + KL(image)
        """
        txt_logits, txt_latent, txt_ib_loss = self.txtclf(txt, mask, segment)
        img_logits, img_latent, img_ib_loss = self.imgclf(img)

        fused_logits = 0.5 * txt_logits + 0.5 * img_logits
        ib_loss = txt_ib_loss + img_ib_loss

        return fused_logits, txt_logits, img_logits, txt_latent, img_latent, ib_loss
