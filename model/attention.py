"""
model/attention.py
==================
Multi-Head Attention — faithful implementation of Vaswani et al. (2017),
Section 3.2.2, equation (4)–(6).

Theory (3 sentences):
    Attention maps a query and a set of key-value pairs to an output.  The
    "scaled dot-product" variant divides QKᵀ by √d_k to counteract vanishing
    gradients that arise when dot products grow large in high dimensions.
    Multi-head attention runs h independent attention functions in parallel on
    linearly projected subspaces, then concatenates and re-projects, letting
    the model jointly attend to information from different representation
    sub-spaces at different positions.

Paper equations:
    Attention(Q,K,V)  = softmax( QKᵀ / √d_k ) · V          (eq. 1)
    head_i            = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
    MultiHead(Q,K,V)  = Concat(head_1,…,head_h) · W^O       (eq. 4/5)
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Attention (Vaswani 2017, Section 3.2.2).

    Parameters
    ----------
    d_model : int
        Model (embedding) dimension.  Must be divisible by `num_heads`.
    num_heads : int
        Number of parallel attention heads (h in the paper).
        Each head operates on a d_h = d_model // num_heads subspace.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, (
            "The dimension of the tokens is not divisible by the number of heads"
        )

        self.num_heads = num_heads
        self.d_h = d_model // num_heads   # d_k = d_v = d_model / h  (paper Section 3.2.2)

        # Learnable projection matrices W^Q, W^K, W^V (each d_model × d_model)
        # The paper uses separate projections per head; equivalently we project
        # the full d_model then reshape — mathematically identical and more efficient.
        self.Q_w = nn.Linear(d_model, d_model)
        self.K_w = nn.Linear(d_model, d_model)
        self.V_w = nn.Linear(d_model, d_model)

        # Output projection W^O (d_model × d_model)
        self.out = nn.Linear(d_model, d_model)

    # ------------------------------------------------------------------
    # Helper: reshape flat embeddings → per-head views
    # ------------------------------------------------------------------

    def create_heads(self, x: Tensor) -> Tensor:
        """
        Split the last dimension into (num_heads, d_h) and transpose so that
        each head's sequence sits on axis -2, enabling batched matmul.

        Shape: (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_h)
        """
        batch, seq_length, d_model = x.shape
        # Reshape: separate d_model into num_heads × d_h
        x = x.view(batch, seq_length, self.num_heads, self.d_h)
        # Move head dimension before seq_length for batched attention matmul
        x = torch.transpose(x, 1, 2)
        return x.contiguous()    # contiguous() needed before any subsequent view()

    # ------------------------------------------------------------------
    # Helper: merge per-head results back to d_model
    # ------------------------------------------------------------------

    def combine_heads(self, x: Tensor) -> Tensor:
        """
        Inverse of `create_heads`: concatenate all head outputs along the
        last axis, recovering a (batch, seq_len, d_model) tensor.

        Shape: (batch, num_heads, seq_len, d_h) → (batch, seq_len, d_model)
        """
        batch, num_heads, seq_length, d_h = x.shape
        return (
            torch.transpose(x, 1, 2)
            .contiguous()
            .view(batch, seq_length, -1)   # -1 = num_heads * d_h = d_model
        )

    # ------------------------------------------------------------------
    # Core: Scaled Dot-Product Attention  (paper eq. 1)
    # ------------------------------------------------------------------

    def scaled_dot_product(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V.

        The 1/√d_k scaling prevents the dot products from entering regions
        where softmax has extremely small gradients (paper Section 3.2.1).

        Parameters
        ----------
        queries, keys, values : Tensor
            Shape (batch, num_heads, seq_len, d_h).
        mask : Optional[Tensor]
            Binary mask where 0 = "ignore this position".  Applied before
            softmax by filling with -1e9 (≈ -∞), which softmax maps to ≈ 0.
            Two mask types are used in practice:
              • Padding mask  (src_mask) — hides <pad> tokens in the encoder.
              • Causal mask   (tgt_mask) — hides future tokens in the decoder
                              (lower-triangular matrix, Section 3.1).

        Returns
        -------
        Tensor
            Attended values, shape (batch, num_heads, seq_len, d_h).
        """
        # QKᵀ — similarity scores; shape: (batch, heads, seq_len_q, seq_len_k)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / (self.d_h ** 0.5)

        # Mask out padding / future positions before softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        # Normalise scores to a probability distribution over the key axis
        weights = torch.softmax(scores, dim=-1)   # (batch, heads, seq_len_q, seq_len_k)

        # Weighted sum of values
        attn_out = torch.matmul(weights, values)  # (batch, heads, seq_len_q, d_h)
        return attn_out

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        Query: Tensor,
        Key: Tensor,
        Value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Full multi-head attention forward pass.

        In the encoder:   Q = K = V = encoder input  (self-attention)
        In the decoder:   Q = decoder state, K = V = encoder output (cross-attention)
                          Q = K = V = decoder input  (masked self-attention)

        Parameters
        ----------
        Query, Key, Value : Tensor
            Shape (batch, seq_len, d_model).
        mask : Optional[Tensor]
            See `scaled_dot_product` for details.

        Returns
        -------
        Tensor
            Shape (batch, seq_len, d_model).
        """
        # Linear projections — each maps d_model → d_model
        Queries = self.create_heads(self.Q_w(Query))   # (batch, heads, seq, d_h)
        Keys    = self.create_heads(self.K_w(Key))     # (batch, heads, seq, d_h)
        Values  = self.create_heads(self.V_w(Value))   # (batch, heads, seq, d_h)

        # Scaled dot-product attention for all heads in parallel
        attentioned_output = self.scaled_dot_product(Queries, Keys, Values, mask)  # (batch, heads, seq, d_h)

        # Merge heads back: (batch, heads, seq, d_h) → (batch, seq, d_model)
        attentioned_output = self.combine_heads(attentioned_output)

        # Final linear projection W^O
        return self.out(attentioned_output)   # (batch, seq_len, d_model)