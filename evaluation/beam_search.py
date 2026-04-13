"""
evaluation/beam_search.py
==========================
Beam search decoder and greedy-decode fallback.

Theory :
    Greedy decoding selects the single highest-probability token at each step,
    which is fast but sub-optimal — a locally good choice may block a globally
    better sequence.  Beam search maintains `beam_size` partial hypotheses at
    each step, expanding each by all vocabulary tokens and keeping only the
    top-k by cumulative log-probability, which finds better translations at
    the cost of O(beam_size × V) work per step.  Length normalisation
    (Wu et al. 2016, adopted by Vaswani 2017) divides the log-probability by
    (5 + len)^α / (5 + 1)^α to prevent the beam from unfairly favouring
    short sequences that accumulate less negative log-probability.
    The paper uses beam_size=4, α=0.6 (Section 6).

Paper reference:
    Section 6 — "We used beam search with a beam size of 4 and length penalty
    α = 0.6."
"""

import logging
from typing import List, Optional

import torch
from torch import Tensor

from model.transformer import Transformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Greedy decode
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(
    model:   Transformer,
    src:     Tensor,
    src_mask: Tensor,
    bos_id:  int,
    eos_id:  int,
    max_len: int  = 200,
    device:  str  = "cuda",
) -> List[List[int]]:
    """
    Greedy auto-regressive decoding for a batch of source sentences.

    At each step, the model predicts a distribution over the target vocab;
    we take the argmax.  Simple, fast, and useful as a training-time
    validation baseline.

    Parameters
    ----------
    model    : Transformer   In eval mode.
    src      : Tensor        (batch, src_len) source token ids.
    src_mask : Tensor        (batch, 1, 1, src_len) source padding mask.
    bos_id   : int           BOS token id.
    eos_id   : int           EOS token id.
    max_len  : int           Maximum number of tokens to generate.
    device   : str

    Returns
    -------
    list of list of int
        Per-sentence list of generated token ids (excluding BOS, up to EOS).
    """
    model.eval()
    batch_size = src.size(0)

    # Run encoder once — reuse for all decoder steps
    src_emb = model.positional_encoding(
        model.embedded_enc(src) * (model.d_model ** 0.5)
    )
    enc_out = src_emb
    for enc_layer in model.encoder_layers:
        enc_out = enc_layer(enc_out, src_mask)

    # Initialise decoder input with BOS for every sentence in the batch
    dec_input = torch.full(
        (batch_size, 1), bos_id, dtype=torch.long, device=device
    )

    # Track which sequences have hit EOS
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    results: List[List[int]] = [[] for _ in range(batch_size)]

    for _ in range(max_len):
        tgt_mask = Transformer.make_tgt_mask(dec_input, pad_idx=0)

        tgt_emb = model.positional_encoding(
            model.embedded_dec(dec_input) * (model.d_model ** 0.5)
        )
        dec_out = tgt_emb
        for dec_layer in model.decoder_layers:
            dec_out = dec_layer(dec_out, enc_out, src_mask, tgt_mask)

        # Logits at the last position only
        logits      = model.out(dec_out[:, -1, :])   # (batch, vocab)
        next_tokens = logits.argmax(dim=-1)           # (batch,)

        # Append predicted token to each incomplete sequence
        for i in range(batch_size):
            if not done[i]:
                tok = next_tokens[i].item()
                if tok == eos_id:
                    done[i] = True
                else:
                    results[i].append(tok)

        if done.all():
            break

        dec_input = torch.cat(
            [dec_input, next_tokens.unsqueeze(1)], dim=1
        )

    return results


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search(
    model:      Transformer,
    src:        Tensor,
    src_mask:   Tensor,
    bos_id:     int,
    eos_id:     int,
    beam_size:  int   = 4,
    alpha:      float = 0.6,
    max_len:    int   = 200,
    device:     str   = "cuda",
) -> List[List[int]]:
    """
    Beam search decoding — paper Section 6.

    Processes one sentence at a time (outer loop over batch) so the beam
    bookkeeping is clean.  For production throughput, batched beam search
    (e.g. fairseq style) would be more efficient, but correctness is more
    important here.

    Length penalty (Wu et al. 2016):
        lp(Y) = (5 + |Y|)^α / (5 + 1)^α

    The final score of a completed hypothesis is:
        score = cumulative_log_prob / lp(Y)

    Parameters
    ----------
    model     : Transformer   Must be in eval mode.
    src       : Tensor        (batch, src_len)
    src_mask  : Tensor        (batch, 1, 1, src_len)
    bos_id    : int
    eos_id    : int
    beam_size : int           Paper: 4.
    alpha     : float         Length penalty exponent.  Paper: 0.6.
    max_len   : int           Hard cap on output length.
    device    : str

    Returns
    -------
    list of list of int
        Best hypothesis token ids for each sentence in the batch.
    """
    model.eval()
    batch_size = src.size(0)
    results:   List[List[int]] = []

    # --- Encode full batch once ---
    import math
    src_emb = model.positional_encoding(
        model.embedded_enc(src) * math.sqrt(model.d_model)
    )
    enc_out = src_emb
    for enc_layer in model.encoder_layers:
        enc_out = enc_layer(enc_out, src_mask)   # (batch, src_len, d_model)

    # --- Decode sentence-by-sentence ---
    for sent_idx in range(batch_size):
        enc_out_i  = enc_out[sent_idx].unsqueeze(0)      # (1, src_len, d_model)
        src_mask_i = src_mask[sent_idx].unsqueeze(0)     # (1, 1, 1, src_len)

        # Each beam is (tokens_so_far, cumulative_log_prob)
        # Start: one beam containing only BOS
        beams: List[tuple] = [([bos_id], 0.0)]
        completed: List[tuple] = []

        for step in range(max_len):
            if not beams:
                break

            # Expand all current beams in one batched forward pass
            all_tokens = [b[0] for b in beams]
            max_t      = max(len(t) for t in all_tokens)

            # Pad to max_t with 0 (we'll only use the last step's logits)
            padded = [t + [0] * (max_t - len(t)) for t in all_tokens]
            dec_input = torch.tensor(padded, dtype=torch.long, device=device)

            tgt_mask = Transformer.make_tgt_mask(dec_input, pad_idx=0)

            # Expand enc_out to match number of active beams
            n_beams    = len(beams)
            enc_rep    = enc_out_i.expand(n_beams, -1, -1)
            mask_rep   = src_mask_i.expand(n_beams, -1, -1, -1)

            tgt_emb = model.positional_encoding(
                model.embedded_dec(dec_input) * math.sqrt(model.d_model)
            )
            dec_out = tgt_emb
            for dec_layer in model.decoder_layers:
                dec_out = dec_layer(dec_out, enc_rep, mask_rep, tgt_mask)

            # Log-probabilities at the final real position for each beam
            last_positions = torch.tensor(
                [len(t) - 1 for t in all_tokens], device=device
            )
            # Gather last real token logits: (n_beams, vocab)
            logits = dec_out[
                torch.arange(n_beams, device=device), last_positions
            ]
            log_probs = torch.log_softmax(logits, dim=-1)   # (n_beams, vocab)

            # Candidate expansion: each beam × vocab
            new_beams: List[tuple] = []
            vocab_size = log_probs.size(-1)

            for b_idx, (tokens, cum_lp) in enumerate(beams):
                # Take top beam_size candidates from this beam's distribution
                top_lps, top_ids = log_probs[b_idx].topk(beam_size)
                for lp, tok in zip(top_lps.tolist(), top_ids.tolist()):
                    new_tokens = tokens + [tok]
                    new_lp     = cum_lp + lp
                    if tok == eos_id:
                        # Completed hypothesis — apply length penalty
                        lp_penalty = ((5 + len(new_tokens)) ** alpha) / (6 ** alpha)
                        completed.append((new_tokens[1:-1], new_lp / lp_penalty))
                    else:
                        new_beams.append((new_tokens, new_lp))

            # Keep only top beam_size active beams (by raw log-prob for now)
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

            # Early exit if we have enough completed hypotheses
            if len(completed) >= beam_size:
                break

        # If no hypothesis completed, take the best active beam
        if not completed:
            if beams:
                best_tokens = beams[0][0][1:]   # strip BOS
                completed   = [(best_tokens, beams[0][1])]
            else:
                completed   = [([], 0.0)]

        # Pick the completed hypothesis with the best length-penalised score
        best = max(completed, key=lambda x: x[1])
        results.append(best[0])

    return results
