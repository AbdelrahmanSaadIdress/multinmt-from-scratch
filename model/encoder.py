"""
model/encoder.py
=================
Encoder stack — Vaswani et al. (2017), Section 3.1 and 3.3.

Theory (4 sentences):
    The encoder maps a source token sequence to a sequence of continuous
    representations z = (z_1, …, z_n) that the decoder can attend to.
    It is a stack of N identical layers, each containing two sub-layers:
    (1) multi-head self-attention over the full source sequence, and
    (2) a position-wise feed-forward network (two linear transforms with ReLU).
    Both sub-layers use a residual connection followed by layer normalisation
    (Pre-LN vs Post-LN: the paper uses Post-LN; this implementation is faithful).

Paper references:
    Encoder stack:         Section 3.1
    Position-wise FFN:     Section 3.3  — FFN(x) = max(0, xW_1+b_1)W_2+b_2
    Residual + LayerNorm:  Section 3.1  — output = LayerNorm(x + Sublayer(x))
    Dropout:               Section 5.4  — applied to sub-layer output before add
"""

import torch.nn as nn
from torch import Tensor
from typing import Optional

from model.attention import MultiHeadAttention


# ---------------------------------------------------------------------------
# Position-Wise Feed-Forward Network  (paper Section 3.3)
# ---------------------------------------------------------------------------

class PositionWiseFeedForward(nn.Module):
    """
    Two-layer position-wise feed-forward network.

    Applied identically and independently to each position — hence "position-
    wise".  The inner dimension d_ff is typically 4× d_model (paper Table 1).

    FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂     (paper eq. 2)

    Parameters
    ----------
    d_model : int
        Input/output dimension.
    d_ff : int
        Inner (hidden) dimension.  Paper default: 2048 (= 4 × 512).
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, X: Tensor) -> Tensor:
        """
        Parameters
        ----------
        X : Tensor
            Shape (batch, seq_len, d_model).

        Returns
        -------
        Tensor
            Same shape as X.
        """
        return self.fc2(self.relu(self.fc1(X)))


# ---------------------------------------------------------------------------
# Single Encoder Layer  (paper Section 3.1)
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """
    One encoder layer consisting of:
      1. Multi-head self-attention
      2. Position-wise feed-forward network
    Each followed by residual addition and layer normalisation.

    Parameters
    ----------
    d_model   : int    Model dimension.
    d_ff      : int    FFN inner dimension.
    num_heads : int    Number of attention heads.
    dropout   : float  Dropout probability (paper Section 5.4).
    """

    def __init__(
        self,
        d_model:   int,
        d_ff:      int,
        num_heads: int,
        dropout:   float,
    ) -> None:
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward   = PositionWiseFeedForward(d_model, d_ff)

        # Two LayerNorm instances — one per sub-layer (paper Section 3.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout applied to sub-layer output *before* the residual add
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        X    : Tensor
            Input, shape (batch, seq_len, d_model).
        mask : Optional[Tensor]
            Padding mask — zeros at <pad> positions so the encoder never
            attends to them.  Shape broadcastable to
            (batch, num_heads, seq_len, seq_len).

        Returns
        -------
        Tensor
            Shape (batch, seq_len, d_model).
        """
        # --- Sub-layer 1: Self-Attention + residual + LayerNorm ---
        # Q = K = V = X  (self-attention: every position attends to all others)
        self_attn_output = self.self_attention(X, X, X, mask)
        X = self.norm1(X + self.dropout(self_attn_output))   # Post-LN (paper faithful)

        # --- Sub-layer 2: Feed-Forward + residual + LayerNorm ---
        feed_forward_output = self.feed_forward(X)
        X = self.norm2(X + self.dropout(feed_forward_output))

        return X   # (batch, seq_len, d_model)
