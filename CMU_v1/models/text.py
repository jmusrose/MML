"""
DML_v1/CMU_v1/models/text.py

TextEncoder：基于 HuggingFace `transformers.BertModel` 的文本编码器。

- 默认加载 `bert-base-uncased`（输出 hidden size 为 768）
- `--freeze_bert` 时冻结 BERT 全部参数
- forward 直接返回 `last_hidden_state`，形状 `(B, L, 768)`
- 模型加载失败抛 `RuntimeError` 并附带提示信息
"""

import torch.nn as nn
from torch import Tensor


class TextEncoder(nn.Module):
    """基于预训练 BERT 的文本编码器。

    Args:
        args: 命名空间对象，至少包含 `bert_model_name` 字段；可选 `freeze_bert`。
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        try:
            from transformers import BertModel
        except ImportError as e:
            raise RuntimeError(
                "Failed to import `transformers.BertModel`. "
                "Please install with `pip install transformers`."
            ) from e

        try:
            self.bert = BertModel.from_pretrained(args.bert_model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load BERT model '{args.bert_model_name}'. "
                f"Please ensure network connectivity, or pre-download the model "
                f"and pass a local path via --bert_model_name. "
                f"Original error: {e}"
            ) from e

        if getattr(args, "freeze_bert", False):
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """前向传播。

        Args:
            input_ids: LongTensor[B, L]
            attention_mask: LongTensor[B, L]，1 = real token, 0 = pad

        Returns:
            last_hidden_state: FloatTensor[B, L, 768]
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state
