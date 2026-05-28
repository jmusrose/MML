#!/usr/bin/env python3
"""
DML Multimodal Classifier for Food-101 classification.

Decision-level fusion: 0.5 * text_logits + 0.5 * image_logits.
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
            txt_features: BERT CLS features [B, 768]
            img_features: flattened image features [B, img_hidden_sz * num_embeds]
        """
        txt_logits, txt_features = self.txtclf(txt, mask, segment)
        img_logits, img_features = self.imgclf(img)

        fused_logits = 0.5 * txt_logits + 0.5 * img_logits

        return fused_logits, txt_logits, img_logits, txt_features, img_features
