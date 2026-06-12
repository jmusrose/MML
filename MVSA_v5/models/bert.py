#!/usr/bin/env python3
"""
BERT text encoder using HuggingFace transformers library.
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BertEncoder(nn.Module):
    """Loads pre-trained BERT model and returns CLS pooled output."""

    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model)

    def forward(self, txt, mask, segment):
        """Returns CLS pooled output: B x 768"""
        outputs = self.bert(
            input_ids=txt,
            token_type_ids=segment,
            attention_mask=mask,
        )
        # pooler_output is the CLS token representation
        return outputs.pooler_output


class BertClf(nn.Module):
    """BERT classifier with RGB_v2-style information bottleneck head."""

    def __init__(self, args):
        super(BertClf, self).__init__()
        self.args = args
        self.enc = BertEncoder(args)
        self.ib_eps_scale = getattr(args, "ib_eps_scale", 1.0)
        self.mu = nn.Linear(args.hidden_sz, args.hidden_sz)
        self.logvar = nn.Linear(args.hidden_sz, args.hidden_sz)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def _sample_features(self, mu, logvar):
        if not self.training or self.ib_eps_scale == 0:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + self.ib_eps_scale * eps * std

    @staticmethod
    def _kl_to_standard_normal(mu, logvar):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=1).mean()

    def forward(self, txt, mask, segment):
        """Returns (logits, sampled_features, kl_loss)."""
        x = self.enc(txt, mask, segment)
        mu = self.mu(x)
        logvar = self.logvar(x)
        latent = self._sample_features(mu, logvar)
        logits = self.clf(latent)
        ib_loss = self._kl_to_standard_normal(mu, logvar)
        return logits, latent, ib_loss
