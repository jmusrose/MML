"""
DML_v1/CMU_v1/models/sequence.py

Sequence encoder for the vision and audio modalities — MIB-style.

This file implements the SAME pipeline used by MIB / Multimodal-Transformer
to encode the vision / acoustic streams::

    (B, T, input_dim)
        --[transpose]--> (B, input_dim, T)
    Conv1d(input_dim -> hidden_sz, kernel_size=k, padding=k//2, bias=False)
        --[NO ReLU / no LayerNorm in between]-->
        --[permute]--> (T, B, hidden_sz)
    MulT TransformerEncoder(layers=num_layers, attn_dropout=0.5,
                            relu_dropout=0.3, res_dropout=0.3,
                            embed_dropout=0.2)
        --[permute back]--> (B, T, hidden_sz)

Notes
-----
- **No activation between Conv and Transformer.** MIB's original code has
  no ReLU / GELU / LayerNorm between the projection conv and the
  Transformer; the previous version of this file inserted a ``nn.ReLU()``
  there, which was empirically zeroing out roughly half of the (z-scored)
  input features and crippled the vision / audio encoders.
- **Conv1d uses ``bias=False``** to match MIB / MulT.
- **Dropout knobs are hard-coded** to MIB's published values
  (0.5 / 0.3 / 0.3 / 0.2). The ``dropout`` constructor argument is kept
  on the signature for backward compatibility but ignored.
- **The ``padding_mask`` argument is accepted but NOT forwarded to the
  Transformer.** This matches MIB's behaviour (full attention; padded
  positions are zero vectors that still attend over real positions).
- The output shape is unchanged: ``(B, T, hidden_sz)``. Pooling (last
  token vs mean pool) is decided by ``Classifier._pool``.
"""

import torch.nn as nn
from torch import Tensor

from models.mult_transformer import TransformerEncoder as MulTTransformerEncoder


class SequenceEncoder(nn.Module):
    """MIB-style 1D-Conv + MulT-Transformer encoder for vision / audio.

    Args:
        input_dim: feature dimension of the raw input sequence (e.g. 47 for
            MOSI vision, 74 for audio, 35 for MOSEI vision).
        hidden_sz: hidden / output dimension. Must be divisible by
            ``num_heads``.
        num_heads: number of attention heads in each Transformer layer.
        num_layers: number of stacked Transformer encoder blocks.
        conv_kernel_size: kernel size of the 1D Conv front-end. Padding is
            set to ``conv_kernel_size // 2`` so the time dimension is
            preserved.
        dropout: kept for backward compatibility; ignored. The Transformer
            uses MIB's published 4-dropout config internally.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sz: int,
        num_heads: int,
        num_layers: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        # 1D conv projection — no bias, no activation (MIB convention)
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_sz,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            bias=False,
        )
        # Xavier init for the projection weights, matching
        # ``MIB._init_custom_parameters``.
        nn.init.xavier_uniform_(self.conv.weight)

        # MulT-style Transformer encoder (pre-norm, 4 independent dropouts,
        # sinusoidal positional embedding, final LayerNorm). Hard-coded
        # dropouts follow MIB's published configuration.
        self.transformer = MulTTransformerEncoder(
            embed_dim=hidden_sz,
            num_heads=num_heads,
            layers=num_layers,
            attn_dropout=0.5,
            relu_dropout=0.3,
            res_dropout=0.3,
            embed_dropout=0.2,
            attn_mask=False,
        )

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        """Encode a batch of variable-length feature sequences.

        Args:
            x: ``FloatTensor[B, T, input_dim]``.
            padding_mask: ``BoolTensor[B, T]`` with ``True`` at padded
                positions. Accepted for API compatibility with the rest
                of the project, but **not forwarded** to the Transformer
                (MIB convention: full attention).

        Returns:
            ``FloatTensor[B, T, hidden_sz]`` — same time dimension as
            input, no internal pooling.
        """
        del padding_mask  # intentionally unused; see module docstring

        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)               # (B, input_dim, T)
        x = self.conv(x)                    # (B, hidden_sz, T)  — no activation
        x = x.permute(2, 0, 1)               # (T, B, hidden_sz) — non-batch-first
        x = self.transformer(x)              # (T, B, hidden_sz)
        x = x.permute(1, 0, 2)               # (B, T, hidden_sz)
        return x
