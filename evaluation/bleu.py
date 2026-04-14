"""
evaluation/bleu.py
===================
BLEU score computation via sacrebleu.

Theory :
    BLEU (Bilingual Evaluation Understudy, Papineni et al. 2002) measures
    n-gram precision between a hypothesis and one or more reference
    translations, multiplied by a brevity penalty to discourage very short
    outputs.  sacrebleu (Post 2018) standardises tokenisation so scores are
    reproducible across papers — raw BLEU varies wildly with tokeniser choice,
    making comparisons meaningless.  The paper (Table 2) reports BLEU using
    the "13a" tokeniser on WMT newstest; we use the same for comparability.

Paper reference: Section 6, Table 2 — evaluation metric.
sacrebleu: https://github.com/mjpost/sacrebleu
"""

import logging
from typing import List, Optional

import sacrebleu
from sacrebleu.metrics import BLEU

logger = logging.getLogger(__name__)

# Sacrebleu tokeniser — "13a" matches the WMT standard used in the paper.
# Use "intl" for Arabic/French if you want better handling of diacritics.
_DEFAULT_TOKENIZE = "13a"

_TOKENIZER_FOR_LANG = {
    "en-ar": "intl",
    "ar-en": "intl",
    "en-fr": "13a",
    "fr-en": "13a",
}

def compute_corpus_bleu(
    hypotheses:  List[str],
    references:  List[List[str]],
    lang_pair:   str = "en-fr",          # ← add this parameter
    tokenize:    Optional[str] = None,
    lowercase:   bool = False,
) -> float:
    """
    Compute corpus-level BLEU for a list of hypotheses against references.

    Parameters
    ----------
    hypotheses : list of str
        Model-generated translations, one per sentence.
    references : list of list of str
        Reference translations.  Outer list = sentences, inner list =
        multiple references per sentence (we typically have 1).
        Shape: [[ref1_sent1, ref2_sent1, ...], [ref1_sent2, ...], ...]
    tokenize : str
        sacrebleu tokeniser.  '13a' = WMT standard; 'intl' = international
        (better for Arabic); 'char' = character-level.
    lowercase : bool
        Lowercase both hypothesis and references before scoring.

    Returns
    -------
    float
        BLEU score in [0, 100].
    """
    if tokenize is None:
        tokenize = _TOKENIZER_FOR_LANG.get(lang_pair, "13a")
        logger.debug("Using sacrebleu tokenizer '%s' for pair '%s'", tokenize, lang_pair)
    
    if not hypotheses:
        logger.warning("compute_corpus_bleu: empty hypotheses list — returning 0.0")
        return 0.0

    # sacrebleu expects references transposed:
    # [[all_refs_for_sent1], [all_refs_for_sent2]] → [[ref1_all_sents], [ref2_all_sents]]
    # i.e. shape (num_refs, num_sents)
    num_refs = max(len(r) for r in references)
    transposed_refs: List[List[str]] = []
    for ref_idx in range(num_refs):
        transposed_refs.append([
            refs[ref_idx] if ref_idx < len(refs) else refs[0]
            for refs in references
        ])

    metric = BLEU(tokenize=tokenize, lowercase=lowercase)
    result = metric.corpus_score(hypotheses, transposed_refs)
    return float(result.score)


def compute_sentence_bleu(
    hypothesis: str,
    references: List[str],
    tokenize:   str = _DEFAULT_TOKENIZE,
) -> float:
    """
    Compute sentence-level BLEU for a single hypothesis.

    Sentence BLEU is noisy and not the primary evaluation metric, but it is
    useful for per-example debugging and for selecting the best beam during
    inference.

    Parameters
    ----------
    hypothesis : str          Single predicted sentence.
    references : list of str  One or more reference strings.
    tokenize   : str          sacrebleu tokeniser.

    Returns
    -------
    float   Sentence BLEU in [0, 100].
    """
    metric = BLEU(tokenize=tokenize, effective_order=True)
    result = metric.sentence_score(hypothesis, references)
    return float(result.score)
