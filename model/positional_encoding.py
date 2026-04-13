"""
model/positional_encoding.py
=============================
Sinusoidal Positional Encoding — Vaswani et al. (2017), Section 3.5.

Theory (3 sentences):
    The Transformer has no recurrence or convolution, so it has no built-in
    notion of token order; positional encodings are added to the embeddings to
    inject sequence position information.  The sinusoidal formulation uses
    fixed (non-learned) sin/cos waves of geometrically increasing periods,
    which the authors hypothesise allows the model to generalise to sequence
    lengths unseen during training.  Each dimension of the encoding corresponds
    to a sinusoid at a different frequency, so positions are represented as
    unique superpositions of waves — analogous to binary counting.

Paper equations (Section 3.5):
    PE(pos, 2i)   = sin( pos / 10000^{2i/d_model} )
    PE(pos, 2i+1) = cos( pos / 10000^{2i/d_model} )
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani 2017, Section 3.5).

    The encoding matrix is computed once at construction time and stored as a
    non-trainable buffer — it moves to the correct device automatically when
    the model is moved (e.g. `.to(device)`).

    Parameters
    ----------
    max_seq_length : int
        Maximum number of positions to pre-compute.  Should match
        `Modelling.max_seq_length` in the config.
    d_model : int
        Embedding dimension.  Encodings have the same dimension as embeddings
        so they can be summed directly.
    """

    def __init__(self, max_seq_length: int, d_model: int) -> None:
        super(PositionalEncoding, self).__init__()

        # --- Build the (max_seq_length, d_model) encoding matrix ---

        pe = torch.zeros(max_seq_length, d_model)

        # Column vector of positions: [0, 1, 2, ..., max_seq_length-1]
        positions = torch.arange(0, max_seq_length, 1).unsqueeze(1)   # (max_seq, 1)

        # Frequency terms: exp(-log(10000) * 2i / d_model)
        # Equivalent to 1 / 10000^{2i/d_model} but numerically more stable via exp/log.
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, d_model, 2) / d_model
        )   # shape: (d_model/2,)

        # Even indices → sine;  Odd indices → cosine  (paper eqs. for PE)
        pe[:, 0::2] = torch.sin(positions * freqs)
        pe[:, 1::2] = torch.cos(positions * freqs)

        # register_buffer: saved in state_dict, moved with the model, not a parameter
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_seq, d_model)

    def forward(self, X: Tensor) -> Tensor:
        """
        Add positional encodings to token embeddings.

        The paper scales embeddings by √d_model before addition (Section 3.4),
        but that scaling is applied in the embedding lookup step inside
        `Transformer.forward`, not here — keeping this module single-purpose.

        Parameters
        ----------
        X : Tensor
            Token embeddings, shape (batch, seq_len, d_model).

        Returns
        -------
        Tensor
            Same shape as X, with positional information added.
        """
        # Slice to actual sequence length (X.size(1) ≤ max_seq_length)
        return X + self.pe[:, :X.size(1), :]
