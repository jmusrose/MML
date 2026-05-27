"""
DML_v1/CMU_v1/models/dml_classifier_mosei.py

MOSEI 数据集对应的 DML 多模态分类器（baseline，决策级融合 / 回归任务）。

设计与 ``models/dml_classifier_mosi.py`` 完全镜像：baseline 阶段不在
MOSI / MOSEI 之间做结构差异化，仅文件归属语义不同（与 RGB_v1 中
``dml_classifier_nyu.py`` / ``dml_classifier_sun.py`` 的同构镜像方式一致）。

设计要点（与 design.md / requirements 5.x 严格一致）：

- 三路独立编码器：
  * `self.vision_enc`  : `SequenceEncoder` (1D Conv + Transformer)
  * `self.audio_enc`   : `SequenceEncoder` (1D Conv + Transformer)
  * `self.text_enc`    : `TextEncoder` (BERT)
  三路参数互不共享。
- 不引入任何介于编码器与回归头之间的中间投影层
  （即没有 unimodal_transform、没有 MLP、没有非线性映射）。
- 三个独立线性回归头（输出维度统一为 `args.n_classes`，baseline 固定为 1）：
  * `self.vision_clf` : `Linear(args.hidden_sz, 1)`
  * `self.audio_clf`  : `Linear(args.hidden_sz, 1)`
  * `self.text_clf`   : `Linear(args.text_hidden_sz, 1)`
- 可切换 PoolStrategy：
  * `last`    : 三路统一取序列最后一个位置 `feat[:, -1, :]`
  * `default` : 文本取 `[CLS]` (`feat[:, 0, :]`)；vision/audio 在有效长度
                 上做 mean pooling（忽略 padding 位置）
- 决策级融合：`both_output = (vision_out + audio_out + text_out) / 3`，
  除此之外不引入任何跨模态交互（无拼接、无注意力、无门控）。
- 返回 7 元组：
  `(both_output, vision_out, audio_out, text_out,
    vision_uni, audio_uni, text_uni)`
"""

import torch.nn as nn
from torch import Tensor

from models.sequence import SequenceEncoder
from models.text import TextEncoder


class Classifier(nn.Module):
    """三模态决策级融合分类器（MOSEI）。

    Args:
        args: 命名空间对象，须包含以下字段：
            - vision_dim, audio_dim
            - hidden_sz, text_hidden_sz
            - num_heads, num_layers, conv_kernel_size, dropout
            - pool_strategy ∈ {"last", "default"}
            - n_classes（baseline 固定为 1）
            - bert_model_name（由 TextEncoder 使用）
            - freeze_bert（可选）
    """

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args

        # 三路独立编码器（参数不共享）
        self.vision_enc = SequenceEncoder(
            input_dim=args.vision_dim,
            hidden_sz=args.hidden_sz,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            conv_kernel_size=args.conv_kernel_size,
            dropout=args.dropout,
        )
        self.audio_enc = SequenceEncoder(
            input_dim=args.audio_dim,
            hidden_sz=args.hidden_sz,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            conv_kernel_size=args.conv_kernel_size,
            dropout=args.dropout,
        )
        self.text_enc = TextEncoder(args)

        # 三路独立线性回归头（无中间投影层）
        out_dim = args.n_classes  # baseline 固定为 1
        self.vision_clf = nn.Linear(args.hidden_sz, out_dim)
        self.audio_clf = nn.Linear(args.hidden_sz, out_dim)
        self.text_clf = nn.Linear(args.text_hidden_sz, out_dim)

    def _pool(self, feat: Tensor, mask: Tensor, modality: str) -> Tensor:
        """根据 ``args.pool_strategy`` 在序列维上做特征聚合。

        Args:
            feat: ``(B, T, D)`` — encoder 输出。
            mask: ``(B, T)`` bool — True 表示 padding 位置。
            modality: ``{"vision", "audio", "text"}``。

        Returns:
            ``(B, D)`` — 聚合后的单模态特征向量。
        """
        if self.args.pool_strategy == "last":
            return feat[:, -1, :]

        # default 策略
        if modality == "text":
            # 文本取 [CLS] token
            return feat[:, 0, :]

        # vision / audio: 在有效长度上 mean pooling，忽略 padding
        valid = (~mask).unsqueeze(-1).float()              # (B, T, 1)
        valid_lens = valid.sum(dim=1).clamp_min(1.0)       # (B, 1)
        return (feat * valid).sum(dim=1) / valid_lens      # (B, D)

    def forward(
        self,
        vision: Tensor,
        vision_mask: Tensor,
        audio: Tensor,
        audio_mask: Tensor,
        text_input_ids: Tensor,
        text_attention_mask: Tensor,
    ):
        """前向传播。

        Args:
            vision: ``FloatTensor[B, T, vision_dim]``
            vision_mask: ``BoolTensor[B, T]``，True = padding
            audio:  ``FloatTensor[B, T, audio_dim]``
            audio_mask:  ``BoolTensor[B, T]``，True = padding
            text_input_ids:      ``LongTensor[B, L]``
            text_attention_mask: ``LongTensor[B, L]``，1 = real, 0 = pad

        Returns:
            7-tuple:
                both_output  : ``(B, 1)`` — 决策级融合后的回归输出
                vision_out   : ``(B, 1)``
                audio_out    : ``(B, 1)``
                text_out     : ``(B, 1)``
                vision_uni   : ``(B, hidden_sz)``
                audio_uni    : ``(B, hidden_sz)``
                text_uni     : ``(B, text_hidden_sz)``
        """
        # 编码器前向：(B, T, hidden_sz) / (B, L, 768)
        v_seq = self.vision_enc(vision, vision_mask)
        a_seq = self.audio_enc(audio, audio_mask)
        t_seq = self.text_enc(text_input_ids, text_attention_mask)

        # BERT 的 attention_mask: 1 = real, 0 = pad；
        # 转换为 SequenceEncoder/Pool 约定的 padding mask: True = pad
        text_pad_mask = (text_attention_mask == 0)

        # Pooling 得到单模态特征向量
        vision_uni = self._pool(v_seq, vision_mask, modality="vision")
        audio_uni = self._pool(a_seq, audio_mask, modality="audio")
        text_uni = self._pool(t_seq, text_pad_mask, modality="text")

        # 单模态回归 logit
        vision_out = self.vision_clf(vision_uni)   # (B, 1)
        audio_out = self.audio_clf(audio_uni)      # (B, 1)
        text_out = self.text_clf(text_uni)         # (B, 1)

        # 决策级融合（唯一融合方式）
        both_output = (vision_out + audio_out + text_out) / 3.0  # (B, 1)

        return (
            both_output,
            vision_out,
            audio_out,
            text_out,
            vision_uni,
            audio_uni,
            text_uni,
        )
