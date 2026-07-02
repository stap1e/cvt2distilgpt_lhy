import re
from collections import Counter
from typing import List, Optional


_NORMAL_TEMPLATES = (
    'no acute cardiopulmonary abnormality',
    'no acute disease',
    'no acute cardiopulmonary disease',
)

_ABNORMAL_KEYWORDS = (
    'opacity',
    'opacities',
    'edema',
    'effusion',
    'pneumonia',
    'pneumothorax',
    'atelectasis',
    'consolidation',
    'cardiomegaly',
    'enlarged',
    'fracture',
    'lesion',
    'mass',
    'nodule',
)


class LiteClinicalRewardEvaluator:
    """
    Lightweight reward scaffold for SCST smoke tests.

    This class intentionally avoids external clinical models. It provides a
    simple text-quality reward that can later be replaced or augmented with
    offline CheXbert/RadGraph/GREEN-style rewards.
    """

    def __init__(
            self,
            min_len: int = 5,
            max_len: int = 120,
            repeat_ngram: int = 3,
            short_penalty: float = 0.15,
            repeat_penalty: float = 0.2,
            normal_template_penalty: float = 0.1,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.repeat_ngram = repeat_ngram
        self.short_penalty = short_penalty
        self.repeat_penalty = repeat_penalty
        self.normal_template_penalty = normal_template_penalty

    def score_one(self, generated: str, reference: Optional[str] = None) -> dict:
        tokens = self._tokens(generated)
        reference_tokens = self._tokens(reference) if reference is not None else []

        repeat_ratio = self._repeat_ngram_ratio(tokens)
        normal_template = self._has_normal_template(generated)

        short = self.short_penalty if len(tokens) < self.min_len else 0.0
        too_long = self.short_penalty if len(tokens) > self.max_len else 0.0
        repeat = self.repeat_penalty * repeat_ratio
        normal = self.normal_template_penalty if self._should_penalize_normal_template(
            normal_template,
            reference_tokens,
            reference,
        ) else 0.0

        penalties = {
            'short': short,
            'too_long': too_long,
            'repeat': repeat,
            'normal_template': normal,
        }
        reward = 1.0 - sum(penalties.values())

        if reference_tokens:
            reward += 0.05 * self._token_f1(tokens, reference_tokens)

        reward = max(0.0, min(1.0, reward))

        return {
            'reward': reward,
            'penalties': penalties,
            'stats': {
                'length': len(tokens),
                'repeat_ngram_ratio': repeat_ratio,
                'normal_template': normal_template,
            },
        }

    def score_batch(self, generated: List[str], references: Optional[List[str]] = None) -> List[dict]:
        if references is None:
            references = [None] * len(generated)
        return [self.score_one(gen, ref) for gen, ref in zip(generated, references)]

    @staticmethod
    def _tokens(text: Optional[str]) -> List[str]:
        if not text:
            return []
        return re.findall(r'[a-z0-9]+', text.lower())

    def _repeat_ngram_ratio(self, tokens: List[str]) -> float:
        if self.repeat_ngram <= 0 or len(tokens) < self.repeat_ngram:
            return 0.0

        ngrams = [tuple(tokens[i:i + self.repeat_ngram]) for i in range(len(tokens) - self.repeat_ngram + 1)]
        counts = Counter(ngrams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        return repeated / max(1, len(ngrams))

    @staticmethod
    def _has_normal_template(text: str) -> bool:
        text = text.lower()
        return any(template in text for template in _NORMAL_TEMPLATES)

    @staticmethod
    def _token_f1(tokens: List[str], reference_tokens: List[str]) -> float:
        if not tokens or not reference_tokens:
            return 0.0

        pred_counts = Counter(tokens)
        ref_counts = Counter(reference_tokens)
        overlap = sum((pred_counts & ref_counts).values())
        if overlap == 0:
            return 0.0

        precision = overlap / len(tokens)
        recall = overlap / len(reference_tokens)
        return 2 * precision * recall / max(precision + recall, 1e-8)

    def _should_penalize_normal_template(
            self,
            normal_template: bool,
            reference_tokens: List[str],
            reference: Optional[str],
    ) -> bool:
        if not normal_template or not reference_tokens:
            return False

        if len(reference_tokens) >= self.min_len * 2:
            return True

        reference_lower = reference.lower() if reference else ''
        return any(keyword in reference_lower for keyword in _ABNORMAL_KEYWORDS)
