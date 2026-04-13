"""
demo/app.py
============
Gradio demo for the multilingual NMT system.

Features:
    • Text input + language-pair selector → translation output
    • Cross-attention heatmap rendered inline (matplotlib → PIL → Gradio Image)
    • Arabic RTL rendering via custom CSS injected into Gradio
    • Beam search decoding (configurable beam size / length penalty)
    • Loads checkpoint from config path; falls back to CPU if no GPU

Arabic RTL note:
    Arabic is right-to-left.  Gradio's default textbox renders LTR.
    We inject `direction: rtl; text-align: right;` CSS onto the output
    textbox when Arabic is the target language.  The attention heatmap
    axis labels are also reversed for Arabic axes (see visualize.py).

Deploy to Hugging Face Spaces:
    1. Push the repo to a HF Space with SDK = "gradio"
    2. Set HF_TOKEN secret if the checkpoint is in a private repo
    3. The Space will auto-install requirements.txt and launch app.py
"""

import io
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Project imports
from data.tokenizer       import MultilingualTokenizer
from evaluation.beam_search import beam_search, greedy_decode
from evaluation.visualize   import extract_cross_attention, plot_attention_heatmap
from model.transformer      import Transformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Language pair definitions
# ---------------------------------------------------------------------------

LANG_PAIRS = [
    ("en", "ar", "English → Arabic"),
    ("en", "fr", "English → French"),
    ("ar", "en", "Arabic → English"),
    ("fr", "en", "French → English"),
    ("ar", "fr", "Arabic → French"),
    ("fr", "ar", "French → Arabic"),
]

PAIR_DISPLAY = [label for _, _, label in LANG_PAIRS]
PAIR_TO_CODES = {label: (src, tgt) for src, tgt, label in LANG_PAIRS}

RTL_LANGS = {"ar"}   # languages that should render right-to-left

# Example sentences for the "Try an example" buttons
EXAMPLES = [
    ["The model learns to translate between languages.", "English → Arabic"],
    ["L'intelligence artificielle transforme le monde.", "French → English"],
    ["الذكاء الاصطناعي يغير العالم.", "Arabic → English"],
    ["The attention mechanism is the core of the Transformer.", "English → French"],
]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class TranslationModel:
    """
    Wrapper that loads a Transformer checkpoint and exposes a translate() method.
    Keeps the model in memory across Gradio requests.
    """

    def __init__(self, checkpoint_path: str, config: dict) -> None:
        self.config    = config
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading model on %s ...", self.device)

        # Load tokenizer
        sp_path = config["Data"]["sp_model_path"]
        self.tokenizer = MultilingualTokenizer(sp_path)

        # Build model architecture
        self.model = Transformer(config).to(self.device)

        # Load weights
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        logger.info("Checkpoint loaded: %s", checkpoint_path)

    def translate(
        self,
        text:      str,
        src_lang:  str,
        tgt_lang:  str,
        beam_size: int   = 4,
        alpha:     float = 0.6,
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Translate a single sentence and return (translation, attention_weights).

        Parameters
        ----------
        text      : str   Source sentence.
        src_lang  : str   Source language code ('en', 'ar', 'fr').
        tgt_lang  : str   Target language code.
        beam_size : int   Beam width.
        alpha     : float Length penalty exponent.

        Returns
        -------
        (translation_str, attn_ndarray or None)
        """
        if not text.strip():
            return "", None

        # Encode source
        src_ids = self.tokenizer.encode(
            text,
            lang=src_lang,
            add_bos=False,
            add_eos=True,
            max_length=self.config["Modelling"]["max_seq_length"],
        )
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        src_mask = Transformer.make_src_mask(src, self.tokenizer.pad_id)

        # Decode
        eval_cfg = self.config["Evaluation"]
        if beam_size > 1:
            pred_ids_list = beam_search(
                model=self.model,
                src=src,
                src_mask=src_mask,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                beam_size=beam_size,
                alpha=alpha,
                max_len=eval_cfg["max_decode_steps"],
                device=self.device,
            )
        else:
            pred_ids_list = greedy_decode(
                model=self.model,
                src=src,
                src_mask=src_mask,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                max_len=eval_cfg["max_decode_steps"],
                device=self.device,
            )

        translation = self.tokenizer.decode(pred_ids_list[0], skip_special_tokens=True)

        # Attention heatmap — encode target to get tgt tensor
        try:
            tgt_ids = self.tokenizer.encode(
                translation, lang=None, add_bos=True, add_eos=True
            )
            tgt = torch.tensor([tgt_ids], dtype=torch.long, device=self.device)
            tgt_mask = Transformer.make_tgt_mask(tgt, self.tokenizer.pad_id)

            attn = extract_cross_attention(
                self.model, src, tgt, src_mask, tgt_mask
            )
        except Exception as exc:
            logger.warning("Could not extract attention: %s", exc)
            attn = None

        return translation, attn


# ---------------------------------------------------------------------------
# Gradio inference function
# ---------------------------------------------------------------------------

def translate_and_visualise(
    source_text:   str,
    language_pair: str,
    beam_size:     int,
    _model_holder: list,   # mutable container so Gradio closures work
) -> Tuple[str, Optional[Image.Image]]:
    """
    Gradio callback: translate source_text and return (translation, heatmap_image).
    """
    model: TranslationModel = _model_holder[0]
    if model is None:
        return "⚠️  Model not loaded — check the checkpoint path in config.", None

    src_lang, tgt_lang = PAIR_TO_CODES[language_pair]

    translation, attn = model.translate(
        source_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        beam_size=beam_size,
    )

    # Build heatmap PIL image if we have attention weights
    heatmap_img = None
    if attn is not None:
        src_pieces = [
            model.tokenizer.id_to_piece(i)
            for i in model.tokenizer.encode(
                source_text, lang=src_lang, add_bos=False, add_eos=True
            )
        ]
        tgt_pieces = [
            model.tokenizer.id_to_piece(i)
            for i in model.tokenizer.encode(
                translation, lang=None, add_bos=True, add_eos=True
            )
        ]
        fig = plot_attention_heatmap(
            attn_weights=attn,
            src_tokens=src_pieces,
            tgt_tokens=tgt_pieces,
            title=f"Cross-attention  {src_lang} → {tgt_lang}",
            rtl_src=(src_lang in RTL_LANGS),
            rtl_tgt=(tgt_lang in RTL_LANGS),
            figsize=(10, 7),
        )
        # Convert matplotlib figure → PIL Image for Gradio
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        heatmap_img = Image.open(buf)

    return translation, heatmap_img


# ---------------------------------------------------------------------------
# RTL CSS helper
# ---------------------------------------------------------------------------

# Injected into Gradio to make Arabic output render right-to-left
RTL_CSS = """
#translation-output textarea {
    direction: rtl;
    text-align: right;
    font-family: 'Noto Sans Arabic', 'Arial Unicode MS', sans-serif;
    font-size: 1.1em;
}
"""

LTR_CSS = """
#translation-output textarea {
    direction: ltr;
    text-align: left;
}
"""


# ---------------------------------------------------------------------------
# Build and launch the Gradio app
# ---------------------------------------------------------------------------

def build_app(config: dict, checkpoint_path: str) -> gr.Blocks:
    """
    Construct the Gradio Blocks interface.

    Parameters
    ----------
    config          : dict   Hydra config (loaded externally).
    checkpoint_path : str    Path to the model checkpoint.

    Returns
    -------
    gr.Blocks   The assembled Gradio app (call .launch() to serve).
    """
    demo_cfg = config.get("Demo", {})

    # Load model once — shared across all requests via closure
    _model_holder: list = [None]
    try:
        _model_holder[0] = TranslationModel(checkpoint_path, config)
    except FileNotFoundError:
        logger.error(
            "Checkpoint not found at '%s'. "
            "Train the model first or update Demo.checkpoint in configs/base.yaml.",
            checkpoint_path,
        )

    with gr.Blocks(
        title=demo_cfg.get("title", "MultiNMT"),
        css=RTL_CSS,   # loaded initially; JS swaps to LTR_CSS for non-Arabic targets
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(f"# 🌐 {demo_cfg.get('title', 'Multilingual NMT')}")
        gr.Markdown(
            demo_cfg.get(
                "description",
                "English ↔ Arabic ↔ French using a faithful Transformer (Vaswani 2017)",
            )
        )

        with gr.Row():
            with gr.Column(scale=2):
                source_box = gr.Textbox(
                    label="Source text",
                    placeholder="Enter text to translate...",
                    lines=4,
                    max_lines=10,
                )
                lang_selector = gr.Dropdown(
                    choices=PAIR_DISPLAY,
                    value=PAIR_DISPLAY[0],
                    label="Translation direction",
                )
                beam_slider = gr.Slider(
                    minimum=1, maximum=8, step=1, value=4,
                    label="Beam size  (1 = greedy)",
                )
                translate_btn = gr.Button("Translate", variant="primary")

            with gr.Column(scale=2):
                output_box = gr.Textbox(
                    label="Translation",
                    lines=4,
                    interactive=False,
                    elem_id="translation-output",
                )
                heatmap_img = gr.Image(
                    label="Cross-attention heatmap",
                    type="pil",
                    interactive=False,
                )

        # Example buttons
        gr.Examples(
            examples=EXAMPLES,
            inputs=[source_box, lang_selector],
            label="Try an example",
        )

        # Wire up the translate button
        translate_btn.click(
            fn=lambda src, lp, beam: translate_and_visualise(
                src, lp, beam, _model_holder
            ),
            inputs=[source_box, lang_selector, beam_slider],
            outputs=[output_box, heatmap_img],
        )

        # Also translate on Enter in the source box
        source_box.submit(
            fn=lambda src, lp, beam: translate_and_visualise(
                src, lp, beam, _model_holder
            ),
            inputs=[source_box, lang_selector, beam_slider],
            outputs=[output_box, heatmap_img],
        )

        gr.Markdown(
            "---\n"
            "**Model**: Transformer (Vaswani et al. 2017) trained on OPUS-100 · "
            "**Tokenizer**: Shared 32k SentencePiece BPE · "
            "[GitHub](https://github.com/your-username/multinmt-from-scratch)"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf

    # Load config via Hydra (or fall back to plain yaml for quick testing)
    try:
        from hydra import compose, initialize
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="base")
        config = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        import yaml
        with open(Path(__file__).parent.parent / "configs" / "base.yaml") as f:
            config = yaml.safe_load(f)

    checkpoint = config["Demo"]["checkpoint"]

    app = build_app(config, checkpoint)
    app.launch(
        share=config["Demo"].get("share", False),
        server_name="0.0.0.0",   # bind to all interfaces for HF Spaces
    )
