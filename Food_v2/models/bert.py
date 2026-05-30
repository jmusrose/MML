#!/usr/bin/env python3
"""
BERT text encoder using HuggingFace transformers library.
"""

import torch.nn as nn
from transformers import BertModel


class BertEncoder(nn.Module):
    """Loads pre-trained BERT model and returns CLS pooled output.

    Output shape: B x 768
    """

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
        return outputs.pooler_output


class BertClf(nn.Module):
    """BERT classifier: BertEncoder + Linear head.

    Maps CLS pooled output (768-dim) to n_classes logits.
    """

    def __init__(self, args):
        super(BertClf, self).__init__()
        self.args = args
        self.enc = BertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, txt, mask, segment):
        """Returns (logits [B, n_classes], features [B, 768])"""
        x = self.enc(txt, mask, segment)
        return self.clf(x), x
