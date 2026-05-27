#!/usr/bin/env python3
"""
BERT text encoder using HuggingFace transformers library.
"""

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
    """BERT classifier: BertEncoder + Linear head."""

    def __init__(self, args):
        super(BertClf, self).__init__()
        self.args = args
        self.enc = BertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, txt, mask, segment):
        """Returns (logits, features)"""
        x = self.enc(txt, mask, segment)
        return self.clf(x), x
