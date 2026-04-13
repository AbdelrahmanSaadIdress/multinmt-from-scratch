"""
model/transformer.py
=====================
Top-level Transformer model — Vaswani et al. (2017), Figure 1.

Theory (4 sentences):
    The full Transformer encoder-decoder maps a source token sequence to
    logits over the target vocabulary at each target position.  The encoder
    builds a context-rich representation of the source; the decoder
    auto-regressively generates the translation by attending to both its
    own output so far and the full encoder output.  Embeddings are shared
    between source and target when the vocabularies are shared (as in this
    multilingual setup), and both embeddings are scaled by √d_model before
    adding positional encodings (paper Section 3.4).  The final linear
    projection + softmax (here just linear — loss handles softmax) maps the
    decoder hidden states to target vocabulary logits.

Paper references:
    Full model:               Figure 1, Section 3
    Embedding scaling:        Section 3.4 — "we multiply those weights by √d_model"
    Shared embeddings:        Section 3.4 (for tied src/tgt vocab)
    Output linear + softmax:  Section 3.4

Multilingual extension (beyond the paper):
    A language tag token (e.g. <ar>) is prepended to every source sentence
    at the data level.  The model learns to condition on it; no architecture
    change is required.  See data/dataset.py for the prepending logic.
"""

import math
import logging
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from model.attention           import MultiHeadAttention           # noqa: F401 (re-exported)
from model.encoder             import EncoderLayer, PositionWiseFeedForward
from model.decoder             import DecoderLayer
from model.positional_encoding import PositionalEncoding

logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer (Vaswani 2017).

    All hyperparameters are read from the config dict so nothing is ever
    hardcoded.  The expected config structure mirrors configs/base.yaml:

        config['Modelling']['d_model']        → int
        config['Modelling']['num_heads']      → int
        config['Modelling']['num_layers']     → int
        config['Modelling']['d_ff']           → int
        config['Modelling']['max_seq_length'] → int
        config['Modelling']['dropout']        → float
        config['Modelling']['src_vocab_size'] → int
        config['Modelling']['tgt_vocab_size'] → int

    Parameters
    ----------
    config : dict
        Hydra / OmegaConf config object (or plain dict for tests).
    """

    def __init__(self, config: dict) -> None:
        super(Transformer, self).__init__()

        # Unpack — all from config, never hardcoded
        src_vocab_size: int   = config['Modelling']['src_vocab_size']
        tgt_vocab_size: int   = config['Modelling']['tgt_vocab_size']
        d_model:        int   = config['Modelling']['d_model']
        num_heads:      int   = config['Modelling']['num_heads']
        num_layers:     int   = config['Modelling']['num_layers']
        d_ff:           int   = config['Modelling']['d_ff']
        max_seq_length: int   = config['Modelling']['max_seq_length']
        dropout:        float = config['Modelling']['dropout']

        # Positional encoding — shared between encoder and decoder embeddings
        self.positional_encoding = PositionalEncoding(max_seq_length, d_model)

        # Token embeddings
        # In this multilingual setup src_vocab_size == tgt_vocab_size (shared BPE vocab).
        # Language tags are regular tokens in this shared vocabulary.
        self.embedded_enc = nn.Embedding(src_vocab_size, d_model)
        self.embedded_dec = nn.Embedding(tgt_vocab_size, d_model)

        # Encoder: N identical layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Decoder: N identical layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final linear projection: d_model → tgt_vocab_size
        # No softmax here — CrossEntropyLoss (or label-smoothed NLL) expects raw logits.
        self.out = nn.Linear(d_model, tgt_vocab_size)

        # d_model stored for embedding scaling (paper Section 3.4)
        self.d_model = d_model

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Transformer initialised — %.2fM trainable parameters", n_params / 1e6)

    # ------------------------------------------------------------------
    # Mask utilities  (called externally by trainer & beam search)
    # ------------------------------------------------------------------

    @staticmethod
    def make_src_mask(src: Tensor, pad_idx: int) -> Tensor:
        """
        Build a padding mask for the source sequence.

        Positions that equal `pad_idx` are masked (set to 0 = "ignore").
        The extra dimensions allow broadcasting over (batch, heads, seq, seq).

        Parameters
        ----------
        src     : Tensor   Shape (batch, src_seq_len) — token ids.
        pad_idx : int      The padding token id from the tokenizer.

        Returns
        -------
        Tensor
            Shape (batch, 1, 1, src_seq_len), dtype bool/uint8.
        """
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_tgt_mask(tgt: Tensor, pad_idx: int) -> Tensor:
        """
        Build a combined padding + causal (look-ahead) mask for the target.

        Two things must be masked:
          1. Padding tokens — same as the source mask.
          2. Future tokens — position i must not see positions j > i.
             Implemented as a lower-triangular matrix (torch.tril).

        Parameters
        ----------
        tgt     : Tensor   Shape (batch, tgt_seq_len) — token ids.
        pad_idx : int      The padding token id.

        Returns
        -------
        Tensor
            Shape (batch, 1, tgt_seq_len, tgt_seq_len).
        """
        tgt_len = tgt.size(1)

        # Causal (look-ahead) mask: lower triangle of 1s
        causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device)
        ).bool()   # (tgt_len, tgt_len)

        # Padding mask for target
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)   # (batch, 1, 1, tgt_len)

        # Combine: a position is valid only if BOTH conditions hold
        return causal_mask.unsqueeze(0) & pad_mask   # (batch, 1, tgt_len, tgt_len)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        src:      Tensor,
        tgt:      Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Full encoder-decoder forward pass (training mode).

        During training the full target sequence (shifted right by 1) is fed
        into the decoder in parallel — teacher forcing.  At inference, tokens
        are generated one at a time; see evaluation/beam_search.py.

        Parameters
        ----------
        src      : Tensor   Source token ids, shape (batch, src_seq_len).
        tgt      : Tensor   Target token ids (teacher-forced), shape (batch, tgt_seq_len).
        src_mask : Optional[Tensor]   Padding mask, shape (batch, 1, 1, src_seq_len).
        tgt_mask : Optional[Tensor]   Causal+padding mask, (batch, 1, tgt_seq_len, tgt_seq_len).

        Returns
        -------
        Tensor
            Logits, shape (batch, tgt_seq_len, tgt_vocab_size).
        """
        # --- Encoder ---
        # Scale embeddings by √d_model before adding positional encodings (Section 3.4)
        src_emb = self.positional_encoding(
            self.embedded_enc(src) * math.sqrt(self.d_model)
        )   # (batch, src_seq_len, d_model)

        enc_out = src_emb
        for encoder in self.encoder_layers:
            enc_out = encoder(enc_out, src_mask)   # (batch, src_seq_len, d_model)

        # --- Decoder ---
        tgt_emb = self.positional_encoding(
            self.embedded_dec(tgt) * math.sqrt(self.d_model)
        )   # (batch, tgt_seq_len, d_model)

        dec_out = tgt_emb
        for decoder in self.decoder_layers:
            dec_out = decoder(dec_out, enc_out, src_mask, tgt_mask)   # (batch, tgt_seq_len, d_model)

        # --- Output projection ---
        return self.out(dec_out)   # (batch, tgt_seq_len, tgt_vocab_size)
