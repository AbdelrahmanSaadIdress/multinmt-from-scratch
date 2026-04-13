"""
training/losses.py
===================
Label-smoothed cross-entropy loss — Vaswani et al. (2017), Section 5.4.

Theory :
    Standard cross-entropy trains the model to assign all probability mass
    to the single correct token, which causes the model to become
    over-confident and poorly calibrated.  Label smoothing (Szegedy et al.
    2016) distributes a small fraction ε of probability mass uniformly across
    all vocabulary tokens, softening the one-hot target into a mixture:
    y_smooth = (1 - ε) · y_one_hot + ε / V.  The paper uses ε = 0.1 and
    reports that this hurts perplexity (the model is less certain) but
    improves BLEU and accuracy because it prevents the model from being
    penalised for placing mass on reasonable alternatives.

Paper reference: Section 5.4
    "We used label smoothing of value ε_ls = 0.1.  This hurts perplexity,
    as the model learns to be more unsure, but improves accuracy and BLEU."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    The smoothed target distribution is:
        q(k) = (1 - ε)  if k == correct token
        q(k) = ε / (V - 1)  otherwise   (uniform over all wrong tokens)

    We implement this efficiently via KL-divergence:
        loss = KL(q_smooth || p_model)
             = -Σ_k q(k) * log p(k)
             = (1-ε) * (-log p(correct)) + ε * mean(-log p(all wrong))

    In practice we use `F.log_softmax` + a direct scatter on the smooth
    distribution, then mask padding tokens so they contribute zero loss.

    Parameters
    ----------
    vocab_size  : int    Target vocabulary size (V).
    pad_idx     : int    Padding token index — excluded from loss computation.
    smoothing   : float  ε, label-smoothing factor.  Paper default: 0.1.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx:    int,
        smoothing:  float = 0.1,
    ) -> None:
        super(LabelSmoothingLoss, self).__init__()

        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing   # mass on the correct token

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute mean label-smoothed cross-entropy over non-padding positions.

        Parameters
        ----------
        logits  : Tensor    Shape (batch * seq_len, vocab_size) — raw model output.
                            Caller should reshape from (batch, seq_len, V) before passing.
        targets : Tensor    Shape (batch * seq_len,) — ground-truth token ids.

        Returns
        -------
        Tensor
            Scalar mean loss over all non-padding positions.
        """
        # Log-probabilities from raw logits
        log_probs = F.log_softmax(logits, dim=-1)   # (N, V)

        # Build smooth target distribution
        # Start with uniform: ε / V everywhere
        smooth_targets = torch.full_like(log_probs, self.smoothing / self.vocab_size)

        # Add remaining confidence to the correct token position
        # scatter_: smooth_targets[i, targets[i]] += confidence - ε/V
        smooth_targets.scatter_(
            dim=1,
            index=targets.unsqueeze(1),
            value=self.confidence + self.smoothing / self.vocab_size,
        )

        # Zero out the padding token column so it contributes no target mass
        smooth_targets[:, self.pad_idx] = 0.0

        # Build padding mask: positions where the target IS padding
        pad_mask = targets.eq(self.pad_idx)   # (N,)

        # KL loss = -Σ q * log p  (sum over vocab, mean over positions)
        loss = -(smooth_targets * log_probs).sum(dim=-1)   # (N,)

        # Zero out loss at padding positions
        loss = loss.masked_fill(pad_mask, 0.0)

        # Normalise by number of non-padding tokens (not batch size)
        # This matches the Fairseq / OpenNMT convention and keeps loss
        # independent of padding ratio.
        n_tokens = (~pad_mask).sum().clamp(min=1)
        return loss.sum() / n_tokens
