"""
data/tokenizer.py
==================
SentencePiece BPE tokenizer wrapper for the multilingual NMT system.

Design decisions (all deliberate):
    1. **Shared vocabulary** — one SentencePiece model trained on all three
       languages (en, ar, fr) jointly.  This encourages the model to share
       representations for cognates and borrowed words (e.g. "café" in en/fr,
       transliterations in ar) and makes zero-shot transfer between language
       pairs possible.

    2. **Language tags as regular tokens** — <en>, <ar>, <fr> are added as
       user-defined symbols in SentencePiece so they get dedicated IDs and are
       never split by BPE.  The tag is prepended to the *source* sentence at
       encode time; the model learns to condition on it.  This is the approach
       used in Johnson et al. (2017) "Google's Multilingual NMT System" and is
       the standard multilingual trick — no architecture change required.

    3. **Arabic note** — SentencePiece handles Arabic script natively.
       `character_coverage=0.9995` (vs 1.0 for pure Latin) is recommended in
       the SentencePiece docs for scripts with large character sets.  The BPE
       merges will produce sub-word units that respect Arabic morphology better
       than character-level, though not as well as a dedicated Arabic
       morphological analyser (beyond-paper improvement: could swap in
       Farasa/CAMeL Tools for pre-tokenisation before SPM training).

References:
    SentencePiece: Kudo & Richardson, 2018  https://arxiv.org/abs/1808.06226
    Multilingual tag:  Johnson et al., 2017  https://arxiv.org/abs/1611.04558
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Special token constants — single source of truth
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

LANG_TOKENS = {
    "en": "<en>",
    "ar": "<ar>",
    "fr": "<fr>",
}

# SentencePiece reserves IDs 0–3 for unk/bos/eos/pad by convention
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


# ---------------------------------------------------------------------------
# Tokenizer class
# ---------------------------------------------------------------------------

class MultilingualTokenizer:
    """
    Thin wrapper around a trained SentencePiece model with utilities for:
        - Training a shared BPE model from raw text files
        - Encoding sentences (with optional language-tag prepending)
        - Decoding token-id sequences back to text
        - Padding / unpadding batches

    Parameters
    ----------
    model_path : str | Path
        Path to a trained `.model` file.  If the file does not yet exist,
        call `MultilingualTokenizer.train(...)` first.
    """

    def __init__(self, model_path: Union[str, Path]) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SentencePiece model not found at '{self.model_path}'. "
                "Run MultilingualTokenizer.train(...) first."
            )

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(self.model_path))

        # Cache the IDs for the tokens we reference frequently
        self.pad_id: int = self._sp.PieceToId(PAD_TOKEN)
        self.bos_id: int = self._sp.PieceToId(BOS_TOKEN)
        self.eos_id: int = self._sp.PieceToId(EOS_TOKEN)
        self.unk_id: int = self._sp.PieceToId(UNK_TOKEN)

        # Language tag IDs — used by the dataset to build language-conditioned src
        self.lang_ids: dict[str, int] = {
            lang: self._sp.PieceToId(token)
            for lang, token in LANG_TOKENS.items()
        }

        # Sanity-check: if any lang tag has UNK id the model wasn't trained correctly
        for lang, tid in self.lang_ids.items():
            if tid == self.unk_id:
                logger.warning(
                    "Language tag '%s' resolved to UNK — did you include it "
                    "in user_defined_symbols during training?", LANG_TOKENS[lang]
                )

        logger.info(
            "Loaded SentencePiece model: vocab_size=%d  pad=%d  bos=%d  eos=%d",
            self.vocab_size, self.pad_id, self.bos_id, self.eos_id,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self._sp.GetPieceSize()

    # ------------------------------------------------------------------
    # Training (class method — call before __init__)
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        input_files: List[Union[str, Path]],
        model_prefix: Union[str, Path],
        vocab_size: int = 32000,
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
        num_threads: int = 8,
    ) -> "MultilingualTokenizer":
        """
        Train a new SentencePiece BPE model on the provided corpus files.

        The model is trained on all input files jointly, producing a single
        shared vocabulary that covers all three languages.  Language tag tokens
        are registered as user_defined_symbols so they receive dedicated IDs
        and are never split by BPE.

        Parameters
        ----------
        input_files : list of str | Path
            Raw text files, one sentence per line (any mix of languages).
        model_prefix : str | Path
            Output path prefix.  SentencePiece writes `{model_prefix}.model`
            and `{model_prefix}.vocab`.
        vocab_size : int
            Shared vocabulary size.  Paper uses 37k (shared src+tgt BPE).
            32k is a sensible default for a 3-language system.
        character_coverage : float
            Fraction of characters covered by the model.  0.9995 is the
            SentencePiece recommendation for scripts beyond Latin.
        model_type : str
            'bpe' (paper) or 'unigram'.
        num_threads : int
            Parallelism for training — increase on multi-core machines.

        Returns
        -------
        MultilingualTokenizer
            A loaded tokenizer backed by the freshly trained model.
        """
        model_prefix = Path(model_prefix)
        model_prefix.parent.mkdir(parents=True, exist_ok=True)

        # Comma-separated list for sentencepiece trainer
        input_str = ",".join(str(f) for f in input_files)

        # Language tags + standard special tokens as user-defined symbols.
        # Listing them here guarantees they get fixed IDs and are never merged.
        user_defined = ",".join(LANG_TOKENS.values())

        # SentencePiece trainer — all options passed as a single string to
        # match the SentencePiece CLI convention.
        train_args = (
            f"--input={input_str} "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} "
            f"--character_coverage={character_coverage} "
            f"--model_type={model_type} "
            f"--num_threads={num_threads} "
            f"--pad_id={PAD_ID} "
            f"--bos_id={BOS_ID} "
            f"--eos_id={EOS_ID} "
            f"--unk_id={UNK_ID} "
            f"--pad_piece={PAD_TOKEN} "
            f"--bos_piece={BOS_TOKEN} "
            f"--eos_piece={EOS_TOKEN} "
            f"--unk_piece={UNK_TOKEN} "
            f"--user_defined_symbols={user_defined} "
            # Shuffle the input corpus before training for balanced coverage
            "--shuffle_input_sentence=true "
            # Keep whitespace normalisation but preserve Arabic/French diacritics
            "--normalization_rule_name=nmt_nfkc_cf "
        )

        logger.info("Training SentencePiece model (vocab=%d, type=%s) ...", vocab_size, model_type)
        spm.SentencePieceTrainer.Train(train_args)
        logger.info("Model saved to '%s.model'", model_prefix)

        return cls(model_prefix.with_suffix(".model"))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        lang: Optional[str] = None,
        add_bos: bool = False,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode a sentence to a list of token IDs.

        The optional `lang` argument prepends the corresponding language tag
        token to the sequence — this is the multilingual conditioning signal
        (Johnson et al. 2017).  The tag is treated as a normal token so it
        participates in attention and gradient flow.

        Parameters
        ----------
        text : str
            Raw sentence in any supported language.
        lang : str | None
            If provided, prepend the language tag token.
            One of {'en', 'ar', 'fr'}.
        add_bos : bool
            Whether to prepend BOS.  (We typically do NOT add BOS to the
            source; the language tag serves the same purpose.)
        add_eos : bool
            Whether to append EOS.  True by default — marks sequence end.
        max_length : int | None
            If set, truncate to this many tokens (including special tokens).

        Returns
        -------
        list of int
            Token IDs.
        """
        ids: List[int] = self._sp.Encode(text, out_type=int)

        # Prepend language tag (before BOS so the model sees: <ar> <s> tokens...)
        if lang is not None:
            if lang not in self.lang_ids:
                raise ValueError(f"Unknown language '{lang}'. Choose from {list(self.lang_ids)}")
            ids = [self.lang_ids[lang]] + ids

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        # Truncate — keep special tokens at the boundaries
        if max_length is not None and len(ids) > max_length:
            # Preserve EOS at the end if we added it
            if add_eos:
                ids = ids[:max_length - 1] + [self.eos_id]
            else:
                ids = ids[:max_length]

        return ids

    def encode_batch(
        self,
        texts: List[str],
        lang: Optional[str] = None,
        add_bos: bool = False,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Encode a list of sentences.  Returns ragged (unpadded) sequences.
        Use `pad_batch` to produce a padded tensor for model input.

        Parameters
        ----------
        texts : list of str
        lang, add_bos, add_eos, max_length : see `encode`

        Returns
        -------
        list of list of int
        """
        return [
            self.encode(t, lang=lang, add_bos=add_bos, add_eos=add_eos,
                        max_length=max_length)
            for t in texts
        ]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back to a string.

        Parameters
        ----------
        ids : list of int
            Token IDs, possibly including padding and special tokens.
        skip_special_tokens : bool
            If True, strip PAD / BOS / EOS / UNK and language tags before
            decoding.  Almost always what you want at inference time.

        Returns
        -------
        str
            Decoded text.
        """
        if skip_special_tokens:
            special = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
            special.update(self.lang_ids.values())
            ids = [i for i in ids if i not in special]

        return self._sp.Decode(ids)

    def decode_batch(
        self, batch_ids: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode a batch of token-ID lists.

        Parameters
        ----------
        batch_ids : list of list of int
        skip_special_tokens : bool

        Returns
        -------
        list of str
        """
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]

    # ------------------------------------------------------------------
    # Padding utilities
    # ------------------------------------------------------------------

    def pad_batch(
        self,
        sequences: List[List[int]],
        max_length: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Right-pad a list of token-ID sequences to the same length.

        Parameters
        ----------
        sequences : list of list of int
            Ragged list of encoded sentences.
        max_length : int | None
            Target length.  Defaults to the length of the longest sequence.

        Returns
        -------
        list of list of int
            All sequences padded to `max_length` with `pad_id`.
        """
        target_len = max_length or max(len(s) for s in sequences)
        return [
            s + [self.pad_id] * (target_len - len(s))
            for s in sequences
        ]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def piece_to_id(self, piece: str) -> int:
        """Return the integer ID for a token string."""
        return self._sp.PieceToId(piece)

    def id_to_piece(self, idx: int) -> str:
        """Return the string piece for an integer ID."""
        return self._sp.IdToPiece(idx)

    def __repr__(self) -> str:
        return (
            f"MultilingualTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"model='{self.model_path.name}')"
        )
