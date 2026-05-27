"""MulT-style TransformerEncoder used by MIB / Multimodal-Transformer.

This is a faithful but compact re-implementation of
``modules.transformer.TransformerEncoder`` from the MulT codebase
(Tsai et al., ACL 2019, https://github.com/yaohungt/Multimodal-Transformer),
which is the Transformer used by MIB to encode the vision / acoustic streams.

Key differences from ``torch.nn.TransformerEncoder``:

- **Pre-norm** layout (LayerNorm before attention and before FFN), with a
  final LayerNorm at the end of the stack.
- **Four independent dropout knobs**:
    * ``embed_dropout``  : applied to ``embed_scale * x + positional_emb``
    * ``attn_dropout``   : passed into ``MultiheadAttention``
    * ``relu_dropout``   : applied between FFN's ReLU and second linear
    * ``res_dropout``    : applied on residual branch (after attn / after FFN)
- **Sinusoidal positional embedding** (added inside, not by the caller).
- I/O follows the original convention: ``(T, B, D)`` (NOT batch-first).

Self-attention only — the cross-modal variant of MulT is intentionally
omitted because MIB's vision / audio streams use only self-attention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding (fairseq-style, no learnable weights)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Lazily instantiated and cached; replaced when a longer sequence appears.
        self.register_buffer("_weights", torch.empty(0), persistent=False)

    @staticmethod
    def _build_table(num_embeddings: int, embedding_dim: int) -> Tensor:
        half_dim = embedding_dim // 2
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb  # (num_embeddings, embedding_dim)

    def forward(self, seq_len: int, batch_size: int, device, dtype) -> Tensor:
        if self._weights.numel() == 0 or self._weights.size(0) < seq_len:
            self._weights = self._build_table(seq_len, self.embedding_dim).to(device=device, dtype=dtype)
        elif self._weights.device != device or self._weights.dtype != dtype:
            self._weights = self._weights.to(device=device, dtype=dtype)
        # Returns (T, B, D) so it can be added directly to a (T, B, D) tensor.
        pe = self._weights[:seq_len].unsqueeze(1).expand(-1, batch_size, -1)
        return pe.detach()


class TransformerEncoderLayer(nn.Module):
    """Single MulT encoder layer (pre-norm, 4 dropout knobs, ReLU FFN)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        res_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout

        # Self-attention (PyTorch's nn.MultiheadAttention is non-batch-first by
        # default, matching MulT's (T, B, D) convention).
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=False,
        )

        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        # Position-wise FFN, fan-out 4x like vanilla Transformer
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (T, B, D)
        # Pre-norm self-attention block
        residual = x
        x = self.attn_layer_norm(x)
        x, _ = self.self_attn(query=x, key=x, value=x, need_weights=False)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x

        # Pre-norm FFN block
        residual = x
        x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        return x


class TransformerEncoder(nn.Module):
    """Stack of MulT-style ``TransformerEncoderLayer`` plus final LayerNorm.

    Args:
        embed_dim: model dimension; must be divisible by ``num_heads``.
        num_heads: number of attention heads.
        layers: number of stacked encoder layers.
        attn_dropout, relu_dropout, res_dropout, embed_dropout: see module
            docstring. Defaults match MIB's hard-coded values for the
            vision / audio Transformer (0.5 / 0.3 / 0.3 / 0.2).
        attn_mask: kept for API parity with the original MulT implementation;
            the self-attention variant ignores it (full attention is used).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        layers: int,
        attn_dropout: float = 0.5,
        relu_dropout: float = 0.3,
        res_dropout: float = 0.3,
        embed_dropout: float = 0.2,
        attn_mask: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.embed_dropout = embed_dropout
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask  # currently unused (self-attention is full)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    relu_dropout=relu_dropout,
                    res_dropout=res_dropout,
                )
                for _ in range(layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: ``FloatTensor[T, B, D]`` (non-batch-first to match MulT).

        Returns:
            ``FloatTensor[T, B, D]`` — last layer output after final LayerNorm.
        """
        T, B, _ = x.shape
        x = self.embed_scale * x
        pe = self.embed_positions(T, B, x.device, x.dtype)  # (T, B, D)
        x = x + pe
        x = F.dropout(x, p=self.embed_dropout, training=self.training)

        for layer in self.layers:
            x = layer(x)

        x = self.layer_norm(x)
        return x
