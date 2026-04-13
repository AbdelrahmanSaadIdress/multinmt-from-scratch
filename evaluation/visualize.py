"""
evaluation/visualize.py
========================
Attention heatmap extraction and rendering.

The cross-attention weights from the final decoder layer are the most
interpretable: they show which source tokens the model attends to when
generating each target token.  These maps are useful for:
    - Diagnosing alignment quality (should look diagonal for similar languages)
    - Detecting copy behaviour (names, numbers)
    - Wandb logging (logged every N steps during training)
    - Gradio demo (shown inline next to translation output)

Arabic note:
    Arabic is right-to-left.  When rendering a heatmap with Arabic on one
    axis, the axis labels should be reversed so they read right-to-left.
    This is handled via the `rtl_src` / `rtl_tgt` flags below.
    In the Gradio demo, use `direction: rtl` CSS on the output text box.
"""

import logging
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers / wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from torch import Tensor

from model.transformer import Transformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attention weight extraction
# ---------------------------------------------------------------------------

def extract_cross_attention(
    model:     Transformer,
    src:       Tensor,
    tgt:       Tensor,
    src_mask:  Tensor,
    tgt_mask:  Tensor,
    layer_idx: int = -1,
    head_idx:  Optional[int] = None,
) -> np.ndarray:
    """
    Run a forward pass in eval mode and return the cross-attention weights
    from a specified decoder layer.

    We hook into the decoder's cross-attention `scaled_dot_product` by
    temporarily monkey-patching it.  This avoids changing the model
    architecture while keeping the extraction clean.

    Parameters
    ----------
    model     : Transformer   In eval mode.
    src       : Tensor        (1, src_len)  — single sentence.
    tgt       : Tensor        (1, tgt_len)
    src_mask  : Tensor        (1, 1, 1, src_len)
    tgt_mask  : Tensor        (1, 1, tgt_len, tgt_len)
    layer_idx : int           Which decoder layer (-1 = last, 0 = first).
    head_idx  : int | None    Which head to return.  None = average all heads.

    Returns
    -------
    np.ndarray
        Shape (tgt_len, src_len).  Entry [i, j] = attention weight from
        target position i to source position j.
    """
    model.eval()
    captured = {}

    # Select the decoder layer we want to probe
    n_layers = len(model.decoder_layers)
    layer    = model.decoder_layers[layer_idx % n_layers]

    # Patch the cross-attention module to capture its weights
    original_sdp = layer.cross_attention.scaled_dot_product

    def capturing_sdp(queries, keys, values, mask=None):
        import math
        scores  = torch.matmul(queries, keys.transpose(-1, -2)) / (layer.cross_attention.d_h ** 0.5)
        if mask is not None:
            scores = torch.masked_fill(scores, mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        captured["weights"] = weights.detach().cpu()   # (1, heads, tgt_len, src_len)
        return torch.matmul(weights, values)

    layer.cross_attention.scaled_dot_product = capturing_sdp

    try:
        with torch.no_grad():
            _ = model(src, tgt, src_mask, tgt_mask)
    finally:
        # Always restore original method, even if forward raises
        layer.cross_attention.scaled_dot_product = original_sdp

    weights = captured["weights"][0]   # (heads, tgt_len, src_len)

    if head_idx is not None:
        attn = weights[head_idx].numpy()
    else:
        attn = weights.mean(dim=0).numpy()   # average over heads

    return attn   # (tgt_len, src_len)


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attn_weights: np.ndarray,
    src_tokens:   List[str],
    tgt_tokens:   List[str],
    title:        str = "Cross-attention",
    rtl_src:      bool = False,
    rtl_tgt:      bool = False,
    figsize:      Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Render an attention weight matrix as a heatmap.

    Parameters
    ----------
    attn_weights : np.ndarray   Shape (tgt_len, src_len).
    src_tokens   : list of str  Source token strings (x-axis).
    tgt_tokens   : list of str  Target token strings (y-axis).
    title        : str          Plot title.
    rtl_src      : bool         Reverse x-axis labels for right-to-left scripts.
    rtl_tgt      : bool         Reverse y-axis labels for right-to-left scripts.
    figsize      : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Trim to actual token lengths (in case of padding)
    tgt_len = min(len(tgt_tokens), attn_weights.shape[0])
    src_len = min(len(src_tokens), attn_weights.shape[1])
    weights = attn_weights[:tgt_len, :src_len]

    im = ax.imshow(weights, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Axis labels
    display_src = list(reversed(src_tokens[:src_len])) if rtl_src else src_tokens[:src_len]
    display_tgt = list(reversed(tgt_tokens[:tgt_len])) if rtl_tgt else tgt_tokens[:tgt_len]

    ax.set_xticks(range(src_len))
    ax.set_xticklabels(display_src, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(tgt_len))
    ax.set_yticklabels(display_tgt, fontsize=9)

    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Target tokens")
    ax.set_title(title)

    fig.tight_layout()
    return fig


def attention_figure_to_numpy(fig: plt.Figure) -> np.ndarray:
    """
    Convert a matplotlib Figure to an HWC uint8 numpy array for wandb logging.

    Parameters
    ----------
    fig : plt.Figure

    Returns
    -------
    np.ndarray   Shape (H, W, 3), dtype uint8.
    """
    fig.canvas.draw()
    w, h   = fig.canvas.get_width_height()
    buf    = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image  = buf.reshape(h, w, 3)
    plt.close(fig)
    return image


def log_attention_to_wandb(
    wandb,
    model:     Transformer,
    src:       Tensor,
    tgt:       Tensor,
    src_mask:  Tensor,
    tgt_mask:  Tensor,
    tokenizer,
    step:      int,
    src_lang:  str = "en",
    tgt_lang:  str = "ar",
) -> None:
    """
    Extract cross-attention, render heatmap, and log to wandb as an image.

    Parameters
    ----------
    wandb      : wandb module
    model      : Transformer
    src        : Tensor   (1, src_len)
    tgt        : Tensor   (1, tgt_len)
    src_mask   : Tensor
    tgt_mask   : Tensor
    tokenizer  : MultilingualTokenizer
    step       : int   Global training step.
    src_lang   : str
    tgt_lang   : str
    """
    try:
        attn = extract_cross_attention(model, src, tgt, src_mask, tgt_mask)

        src_tokens = [tokenizer.id_to_piece(i) for i in src[0].tolist()]
        tgt_tokens = [tokenizer.id_to_piece(i) for i in tgt[0].tolist()]

        fig = plot_attention_heatmap(
            attn_weights=attn,
            src_tokens=src_tokens,
            tgt_tokens=tgt_tokens,
            title=f"Cross-attention  {src_lang}→{tgt_lang}  step {step}",
            rtl_src=(src_lang == "ar"),
            rtl_tgt=(tgt_lang == "ar"),
        )

        img = attention_figure_to_numpy(fig)
        wandb.log({"attention/heatmap": wandb.Image(img)}, step=step)

    except Exception as exc:
        logger.warning("Failed to log attention heatmap: %s", exc)
