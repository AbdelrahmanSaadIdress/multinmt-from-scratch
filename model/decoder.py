"""
model/decoder.py
=================
Decoder stack — Vaswani et al. (2017), Section 3.1.

Theory (4 sentences):
    The decoder auto-regressively generates the output sequence one token at a
    time, conditioning on both the encoder output (source context) and the
    previously generated target tokens.  Each decoder layer has THREE sub-
    layers: (1) masked self-attention over the target prefix — the causal mask
    prevents position i from attending to positions j > i, preserving the
    auto-regressive property; (2) cross-attention where queries come from the
    decoder and keys/values come from the encoder — this is how the decoder
    "reads" the source; (3) the same position-wise FFN as the encoder.
    All three sub-layers use the same Post-LN residual pattern.

Paper references:
    Decoder stack:         Section 3.1
    Masked self-attention: Section 3.1 — "prevent positions from attending to
                           subsequent positions"
    Cross-attention:       Section 3.2.3 — encoder-decoder attention
"""

import torch.nn as nn
from torch import Tensor
from typing import Optional

from model.attention import MultiHeadAttention
from model.encoder import PositionWiseFeedForward


# ---------------------------------------------------------------------------
# Single Decoder Layer  (paper Section 3.1)
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """
    One decoder layer consisting of:
      1. Masked multi-head self-attention  (causal / look-ahead mask)
      2. Multi-head cross-attention        (attends to encoder output)
      3. Position-wise feed-forward network
    Each followed by residual addition and layer normalisation.

    Parameters
    ----------
    d_model   : int    Model dimension.
    d_ff      : int    FFN inner dimension.
    num_heads : int    Number of attention heads.
    dropout   : float  Dropout probability.
    """

    def __init__(
        self,
        d_model:   int,
        d_ff:      int,
        num_heads: int,
        dropout:   float,
    ) -> None:
        super(DecoderLayer, self).__init__()

        # Sub-layer 1: causal self-attention (target tokens attend to earlier target tokens)
        self.self_attention  = MultiHeadAttention(d_model, num_heads)

        # Sub-layer 2: cross-attention (target queries attend to full encoder output)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        # Sub-layer 3: position-wise feed-forward
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        # Three LayerNorm instances — one per sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        X:        Tensor,
        enc_out:  Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        X        : Tensor
            Target embeddings, shape (batch, tgt_seq_len, d_model).
        enc_out  : Tensor
            Final encoder output, shape (batch, src_seq_len, d_model).
            Used as keys and values in cross-attention.
        src_mask : Optional[Tensor]
            Padding mask for the source sequence — prevents cross-attention
            from attending to <pad> tokens in the encoder output.
        tgt_mask : Optional[Tensor]
            Causal mask for the target sequence — lower-triangular matrix
            that prevents each position from seeing future tokens.

        Returns
        -------
        Tensor
            Shape (batch, tgt_seq_len, d_model).
        """
        # --- Sub-layer 1: Masked Self-Attention (causal) ---
        # Q = K = V = X;  tgt_mask hides future positions
        self_attn_output = self.self_attention(X, X, X, tgt_mask)
        X = self.norm1(X + self.dropout(self_attn_output))

        # --- Sub-layer 2: Cross-Attention (encoder-decoder) ---
        # Queries come from the decoder; Keys and Values from the encoder output.
        # This is the mechanism by which the decoder reads source context.
        cross_attn_output = self.cross_attention(X, enc_out, enc_out, src_mask)
        X = self.norm2(X + self.dropout(cross_attn_output))

        # --- Sub-layer 3: Feed-Forward ---
        feed_forward_output = self.feed_forward(X)
        X = self.norm3(X + self.dropout(feed_forward_output))

        return X   # (batch, tgt_seq_len, d_model)
