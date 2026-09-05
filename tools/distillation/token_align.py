"""Cross-tokenizer token alignment for teacher log-prob transfer.

The student (GPT2 BPE, 50257) and the teacher (e.g. MedGemma/Qwen SentencePiece)
use different vocabularies and token boundaries, so a per-token KL on the full
next-token distribution is impossible. What we CAN transfer offline is the
teacher's log-probability of the tokens the student actually emitted. This
module maps teacher token log-probs onto student tokens through character
spans, which both fast tokenizers can expose.

Pure Python; no torch/transformers imports at module level.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def spans_from_offsets(
    offsets: Sequence[Sequence[int]],
) -> List[Tuple[int, int]]:
    """Convert HF ``return_offsets_mapping`` output to (start, end) char spans.

    Zero-length spans (special tokens) are dropped so alignment only sees
    characters of the real text.
    """
    spans = []
    for pair in offsets:
        if len(pair) != 2:
            raise ValueError(f"offset pair must be [start, end], got {pair!r}")
        start, end = int(pair[0]), int(pair[1])
        if end > start:
            spans.append((start, end))
    return spans


def _overlap(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> int:
    return max(0, min(span_a[1], span_b[1]) - max(span_a[0], span_b[0]))


def _center(span: Tuple[int, int]) -> float:
    return 0.5 * (span[0] + span[1])


def align_teacher_to_student(
    student_spans: Sequence[Tuple[int, int]],
    teacher_spans: Sequence[Tuple[int, int]],
    teacher_logps: Sequence[float],
    unmatched: float = 0.0,
) -> Tuple[List[float], float]:
    """Assign each student token a teacher score.

    A student token receives the mean teacher log-prob over teacher tokens
    whose char span overlaps it. If no teacher token overlaps (different
    pre-tokenization of whitespace/punctuation), the nearest teacher token by
    span-center distance is used. Empty teacher side yields ``unmatched`` for
    every student token.

    Returns:
        (per-student-token scores, coverage in [0, 1]) where coverage is the
        fraction of student tokens that received an overlapping (not
        nearest-neighbour fallback) teacher score.
    """
    if len(teacher_spans) != len(teacher_logps):
        raise ValueError(
            "teacher_spans and teacher_logps must have equal length, got "
            f"{len(teacher_spans)} != {len(teacher_logps)}"
        )

    if not teacher_spans:
        return [unmatched] * len(student_spans), 0.0

    scores: List[float] = []
    matched = 0
    for student_span in student_spans:
        overlapping = [
            (span, logp)
            for span, logp in zip(teacher_spans, teacher_logps)
            if _overlap(student_span, span) > 0
        ]
        if overlapping:
            matched += 1
            scores.append(sum(logp for _, logp in overlapping) / len(overlapping))
            continue

        nearest_span = min(
            teacher_spans,
            key=lambda span: abs(_center(span) - _center(student_span)),
        )
        index = teacher_spans.index(nearest_span)
        scores.append(float(teacher_logps[index]))

    coverage = matched / max(1, len(student_spans))
    return scores, coverage


def aligned_length_mismatch(
    student_token_ids: Sequence[int],
    scores: Sequence[float],
) -> bool:
    """True when the stored score array no longer matches the token ids."""
    return len(student_token_ids) != len(scores)
