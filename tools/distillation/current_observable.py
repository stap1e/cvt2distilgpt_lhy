"""Current-observable report filtering (extracted from COVAR-V2, torch-free).

The class body is identical to the previous in-model definition in
``cvt2distilgpt2_mimic_cxr_visual_grounded_v2.py``; it was moved here so that
offline teacher tooling (Form A rewriting) can reuse exactly the same filter
without importing torch/lightning. The model module re-imports it, so
``CurrentObservableReportFilterV2`` keeps working for checkpoints and
training code.
"""

from __future__ import annotations

import re
from typing import Optional

COVAR_V2_VERSION = "2026-07-14-covar-v2"


class CurrentObservableReportFilterV2:
    """Distil single-image-observable statements from a MIMIC-CXR report.

    The filter is used on TRAINING targets only. Validation and test references
    remain unchanged, so standard MIMIC-CXR metrics are still calculated.
    """

    _split_re = re.compile(r"(?<=[.!?])\s+|\n+")
    _leading_comparison_re = re.compile(
        r"^(?:as\s+)?(?:compared\s+(?:to|with)|in\s+comparison\s+(?:to|with))"
        r"[^,.;:]*[,;:]\s*",
        re.IGNORECASE,
    )
    _temporal_re = re.compile(
        r"\b(?:"
        r"compared\s+(?:to|with)|comparison\s+(?:to|with)|previous|prior|"
        r"interval(?:\s+(?:change|development))?|unchanged|stable|"
        r"no\s+(?:relevant|significant)?\s*change|improved|worsened|"
        r"increased\s+since|decreased\s+since|new(?:\s+since)?|"
        r"again\s+(?:seen|noted)|since\s+the\s+(?:last|previous|prior)|"
        r"has\s+been\s+(?:extubated|intubated|removed|placed|inserted|advanced|retracted)|"
        r"was\s+(?:removed|placed|inserted|advanced|retracted)"
        r")\b",
        re.IGNORECASE,
    )

    _device_rewrites = (
        (
            re.compile(r"\b(?:the\s+patient\s+)?has\s+been\s+extubated\b", re.I),
            "the endotracheal tube is not present",
        ),
        (
            re.compile(r"\b(?:the\s+patient\s+)?has\s+been\s+intubated\b", re.I),
            "an endotracheal tube is present",
        ),
        (
            re.compile(
                r"\b(?:the\s+)?(?:nasogastric|enteric|feeding)\s+tube\s+"
                r"(?:has\s+been|was)\s+removed\b",
                re.I,
            ),
            "the enteric tube is not present",
        ),
        (
            re.compile(
                r"\b(?:the\s+)?(?:nasogastric|enteric|feeding)\s+tube\s+"
                r"(?:has\s+been|was)\s+(?:placed|inserted)\b",
                re.I,
            ),
            "an enteric tube is present",
        ),
        (
            re.compile(
                r"\b(?:the\s+patient\s+)?has\s+received\s+(?:a\s+)?"
                r"(?:nasogastric|enteric|feeding)\s+tube\b",
                re.I,
            ),
            "an enteric tube is present",
        ),
        (
            re.compile(
                r"\b(?:the\s+)?chest\s+tube\s+(?:has\s+been|was)\s+removed\b",
                re.I,
            ),
            "the chest tube is not present",
        ),
        (
            re.compile(
                r"\b(?:the\s+)?(?:central\s+(?:venous\s+)?(?:line|catheter)|picc)\s+"
                r"(?:has\s+been|was)\s+(?:placed|inserted)\b",
                re.I,
            ),
            "a central venous catheter is present",
        ),
    )

    def __init__(self, fallback: str = "keep_original", min_words: int = 3):
        if fallback not in {"keep_original", "keep_longest"}:
            raise ValueError(
                "temporal_filter_fallback must be 'keep_original' or 'keep_longest'."
            )
        self.fallback = fallback
        self.min_words = int(min_words)

    @staticmethod
    def _normalise_sentence(sentence: str) -> str:
        sentence = re.sub(r"\s+", " ", sentence).strip(" \t\r\n.;")
        return sentence + "." if sentence else ""

    def _rewrite_device_transition(self, sentence: str) -> Optional[str]:
        rewritten = sentence
        changed = False
        for pattern, replacement in self._device_rewrites:
            if pattern.search(rewritten):
                rewritten = pattern.sub(replacement, rewritten)
                changed = True
        if not changed:
            return None
        rewritten = self._leading_comparison_re.sub("", rewritten)
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        if self._temporal_re.search(rewritten):
            return None
        return self._normalise_sentence(rewritten)

    def filter_report(self, report: str):
        raw = re.sub(r"\s+", " ", str(report)).strip()
        if not raw:
            return raw, {"removed": 0, "rewritten": 0, "fallback": 0}

        sentences = [s.strip() for s in self._split_re.split(raw) if s.strip()]
        if not sentences:
            sentences = [raw]

        kept = []
        removed = 0
        rewritten_count = 0

        for sentence in sentences:
            without_prefix = self._leading_comparison_re.sub("", sentence).strip()
            if (
                without_prefix != sentence
                and without_prefix
                and not self._temporal_re.search(without_prefix)
            ):
                kept.append(self._normalise_sentence(without_prefix))
                rewritten_count += 1
                continue

            device_state = self._rewrite_device_transition(sentence)
            if device_state is not None:
                kept.append(device_state)
                rewritten_count += 1
                continue

            if self._temporal_re.search(sentence):
                removed += 1
                continue

            normalised = self._normalise_sentence(sentence)
            if normalised:
                kept.append(normalised)

        filtered = " ".join(kept).strip()
        if len(filtered.split()) >= self.min_words:
            return filtered, {
                "removed": removed,
                "rewritten": rewritten_count,
                "fallback": 0,
            }

        fallback_text = (
            max(sentences, key=lambda x: len(x.split()))
            if self.fallback == "keep_longest"
            else raw
        )
        return fallback_text, {
            "removed": removed,
            "rewritten": rewritten_count,
            "fallback": 1,
        }

    def contains_temporal_claim(self, report: str) -> bool:
        return bool(self._temporal_re.search(str(report)))
