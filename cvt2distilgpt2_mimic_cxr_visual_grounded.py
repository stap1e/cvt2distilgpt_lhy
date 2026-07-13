"""
Visual-dependence regularised CvT2DistilGPT2 for MIMIC-CXR.

Validation diagnostic fix: 2026-07-13-v2.

This module is intentionally implemented as a subclass of the original
CvT2DistilGPT2MIMICXRChen model so that the existing dataset, metrics,
checkpoint layout, DPO/SCST utilities, and generation code remain usable.

The first-stage method does not require anatomy boxes, RadGraph, Chest
ImaGenome, or a new dataset file. It adds three forms of supervision:

1. Null-image clinical-token margin:
   ground-truth clinical tokens should receive higher probability with the
   real image than with an all-zero visual memory.
2. Mismatched-image clinical-token margin:
   the real image should outperform a report-dissimilar image selected from
   the same mini-batch.
3. Visual concept-state prediction:
   pooled image tokens predict positive/negative/uncertain states weakly
   extracted from the paired report. This creates a direct visual-only path
   to clinical semantics instead of relying exclusively on GPT language loss.

The regularisers are restricted to clinical phrase spans. Function words are
optionally kept invariant between the real-image and null-image branches.

Recommended first experiment:
    train_mode: ce
    null_margin_weight: 0.20
    mismatch_margin_weight: 0.20
    concept_aux_weight: 0.10
    nonclinical_invariance_weight: 0.01

The code is a dependency-light first version. A later version can replace the
lexical phrase matcher with RadGraph claims and random/mismatched negatives
with anatomy-specific counterfactual region interventions.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from cvt2distilgpt2_mimic_cxr_chen import CvT2DistilGPT2MIMICXRChen
from tools.preference_rl import build_decoder_lm_batch


VALIDATION_DIAGNOSTIC_FIX_VERSION = "2026-07-13-v2"


# The aliases are deliberately radiology-oriented and relatively conservative.
# They are used both for exact token-span matching and weak concept-state labels.
DEFAULT_CLINICAL_CONCEPTS: Mapping[str, Sequence[str]] = {
    "atelectasis": ("atelectasis", "atelectatic opacity", "subsegmental atelectasis"),
    "cardiomegaly": ("cardiomegaly", "enlarged cardiac silhouette", "cardiac enlargement"),
    "consolidation": ("consolidation", "airspace consolidation", "focal consolidation"),
    "edema": ("pulmonary edema", "interstitial edema", "vascular congestion", "pulmonary vascular congestion"),
    "effusion": ("pleural effusion", "pleural effusions"),
    "emphysema": ("emphysema", "emphysematous change", "hyperinflation"),
    "fibrosis": ("fibrosis", "fibrotic change", "chronic interstitial change"),
    "fracture": ("fracture", "rib fracture", "osseous injury"),
    "hernia": ("hiatal hernia", "hernia"),
    "infiltrate": ("infiltrate", "infiltration", "pulmonary infiltrate"),
    "lung_lesion": ("lung lesion", "pulmonary lesion", "masslike opacity"),
    "lung_opacity": ("lung opacity", "pulmonary opacity", "airspace opacity", "airspace opacities", "focal opacity"),
    "mass": ("lung mass", "pulmonary mass", "mediastinal mass"),
    "nodule": ("pulmonary nodule", "lung nodule", "nodular opacity"),
    "pneumonia": ("pneumonia", "bronchopneumonia"),
    "pneumothorax": ("pneumothorax", "apical pneumothorax"),
    "pleural_abnormality": ("pleural thickening", "pleural abnormality", "pleural disease"),
    "support_device": (
        "endotracheal tube", "enteric tube", "feeding tube", "nasogastric tube",
        "central venous catheter", "central line", "picc", "pacemaker",
        "defibrillator", "chest tube", "support device", "support devices",
    ),
    "mediastinal_widening": ("widened mediastinum", "mediastinal widening"),
    "low_lung_volume": ("low lung volume", "low lung volumes", "hypoinflation"),
    "clear_lungs": ("lungs are clear", "clear lungs", "no focal airspace disease"),
    "acute_process": (
        "acute cardiopulmonary process", "acute cardiopulmonary abnormality",
        "acute intrathoracic process", "acute pulmonary process",
    ),
}

NEGATION_RE = re.compile(
    r"(?:\bno\b|\bwithout\b|\bnegative for\b|\babsence of\b|\bfree of\b|"
    r"\bnot seen\b|\bnot identified\b|\bno evidence of\b|\bno definite\b)"
    r"(?:\W+\w+){0,6}\W*$",
    re.IGNORECASE,
)
UNCERTAINTY_RE = re.compile(
    r"(?:\bpossible\b|\bpossibly\b|\bprobable\b|\blikely\b|\bmay represent\b|"
    r"\bcannot exclude\b|\bcould represent\b|\bsuspicious for\b|\bquestion of\b)"
    r"(?:\W+\w+){0,6}\W*$",
    re.IGNORECASE,
)

# These phrases affect the factual interpretation of a finding but are not
# suitable as independent disease-classification targets.
TOKEN_ONLY_CLINICAL_PHRASES: Tuple[str, ...] = (
    "right", "left", "bilateral", "unilateral",
    "upper lobe", "middle lobe", "lower lobe", "lung base", "lung bases",
    "apical", "bibasilar", "perihilar", "retrocardiac", "subpleural",
    "small", "trace", "mild", "moderate", "large", "severe",
    "increased", "decreased", "improved", "worsened", "stable", "unchanged",
)


class ClinicalPhraseSupervisor:
    """Build token-span masks and weak report-derived concept-state targets."""

    NEGATIVE = 0
    POSITIVE = 1
    UNCERTAIN = 2
    IGNORE_INDEX = -100

    def __init__(
        self,
        tokenizer,
        concepts: Mapping[str, Sequence[str]] = DEFAULT_CLINICAL_CONCEPTS,
    ) -> None:
        self.tokenizer = tokenizer
        self.concept_names = tuple(concepts.keys())
        self.aliases: Dict[str, Tuple[str, ...]] = {
            name: tuple(dict.fromkeys(alias.lower().strip() for alias in aliases if alias.strip()))
            for name, aliases in concepts.items()
        }

        # Exact GPT2 token patterns. GPT2 uses a distinct tokenisation after a
        # leading space, therefore both variants are retained.
        token_patterns: List[Tuple[int, ...]] = []
        assertion_prefixes = (
            "", "no ", "without ", "no evidence of ", "possible ",
            "possibly ", "likely ", "cannot exclude ", "may represent ",
        )
        token_phrases: List[str] = list(TOKEN_ONLY_CLINICAL_PHRASES)
        for aliases in self.aliases.values():
            for alias in aliases:
                token_phrases.extend(prefix + alias for prefix in assertion_prefixes)

        for phrase in token_phrases:
            for text in (phrase, " " + phrase):
                ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
                if ids:
                    token_patterns.append(ids)
        # Longest phrases first makes diagnostics easier and avoids needless scans.
        self.token_patterns = tuple(sorted(set(token_patterns), key=len, reverse=True))

        self.alias_patterns: Dict[str, Tuple[re.Pattern, ...]] = {}
        for name, aliases in self.aliases.items():
            patterns = []
            for alias in aliases:
                # Flexible whitespace but strict word boundaries at both ends.
                escaped = re.escape(alias).replace(r"\ ", r"\s+")
                patterns.append(re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE))
            self.alias_patterns[name] = tuple(patterns)

    @property
    def num_concepts(self) -> int:
        return len(self.concept_names)

    def clinical_token_mask(
        self,
        label_ids: torch.Tensor,
        pad_token_id: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Return [B, L] bool mask covering exact clinical phrase spans."""
        if label_ids.ndim != 2:
            raise ValueError(f"label_ids must be [B, L], got {tuple(label_ids.shape)}")

        mask = torch.zeros_like(label_ids, dtype=torch.bool)
        sequences = label_ids.detach().cpu().tolist()
        for batch_index, sequence in enumerate(sequences):
            valid_length = len(sequence)
            for i, token_id in enumerate(sequence):
                if token_id == pad_token_id or (eos_token_id is not None and token_id == eos_token_id):
                    valid_length = i
                    break
            sequence = sequence[:valid_length]
            for pattern in self.token_patterns:
                width = len(pattern)
                if width > valid_length:
                    continue
                for start in range(valid_length - width + 1):
                    if tuple(sequence[start:start + width]) == pattern:
                        mask[batch_index, start:start + width] = True
        return mask

    @staticmethod
    def _mention_state(text: str, match_start: int) -> int:
        prefix = text[max(0, match_start - 120):match_start]
        if NEGATION_RE.search(prefix):
            return ClinicalPhraseSupervisor.NEGATIVE
        if UNCERTAINTY_RE.search(prefix):
            return ClinicalPhraseSupervisor.UNCERTAIN
        return ClinicalPhraseSupervisor.POSITIVE

    def concept_state_targets(self, reports: Sequence[str], device: torch.device) -> torch.Tensor:
        """
        Extract [B, C] labels in {negative, positive, uncertain, ignore}.

        When a concept is mentioned multiple times, positive dominates uncertain,
        and uncertain dominates negative. This avoids treating a final positive
        statement as negative merely because an earlier differential was negated.
        """
        targets = torch.full(
            (len(reports), self.num_concepts),
            fill_value=self.IGNORE_INDEX,
            dtype=torch.long,
            device=device,
        )
        priority = {
            self.NEGATIVE: 0,
            self.UNCERTAIN: 1,
            self.POSITIVE: 2,
        }
        for batch_index, raw_report in enumerate(reports):
            report = str(raw_report).lower()
            for concept_index, concept_name in enumerate(self.concept_names):
                best_state: Optional[int] = None
                for pattern in self.alias_patterns[concept_name]:
                    for match in pattern.finditer(report):
                        state = self._mention_state(report, match.start())
                        if best_state is None or priority[state] > priority[best_state]:
                            best_state = state
                if best_state is not None:
                    targets[batch_index, concept_index] = best_state
        return targets


def _safe_target_log_probs(
    logits: torch.Tensor,
    label_ids: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return target-token log probabilities and a valid-token mask."""
    if logits.shape[:2] != label_ids.shape:
        raise ValueError(
            f"logits/labels length mismatch: logits={tuple(logits.shape)}, "
            f"labels={tuple(label_ids.shape)}"
        )
    valid = label_ids.ne(pad_token_id)
    safe_labels = label_ids.masked_fill(~valid, 0)
    # Compute log-softmax in fp32 for numerical stability under mixed precision.
    log_probs = F.log_softmax(logits.float(), dim=-1)
    target_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return target_log_probs, valid


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    denominator = mask.sum()
    if denominator.item() == 0:
        # Preserve graph/device/dtype while returning an exact zero.
        return values.sum() * 0.0
    return (values * mask).sum() / denominator


class CvT2DistilGPT2MIMICXRVisualGrounded(CvT2DistilGPT2MIMICXRChen):
    """CvT2DistilGPT2 with clinical-token visual-dependence regularisation."""

    def __init__(
        self,
        *args,
        null_margin_weight: float = 0.20,
        mismatch_margin_weight: float = 0.20,
        visual_margin: float = 0.20,
        nonclinical_invariance_weight: float = 0.01,
        concept_aux_weight: float = 0.10,
        concept_head_dropout: float = 0.10,
        detach_mismatch_features: bool = True,
        grounding_mask_fallback: str = "none",
        log_grounding_diagnostics: bool = True,
        strict_grounding_diagnostics: bool = False,
        generation_no_repeat_ngram_size: int = 3,
        generation_repetition_penalty: float = 1.10,
        generation_length_penalty: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if self.train_mode != "ce":
            raise ValueError(
                "The visual-grounded first version currently supports train_mode='ce' only. "
                "DPO/SCST can be reintroduced after the CE visual-dependence stage."
            )
        if grounding_mask_fallback not in {"none", "valid"}:
            raise ValueError("grounding_mask_fallback must be 'none' or 'valid'.")
        for name, value in {
            "null_margin_weight": null_margin_weight,
            "mismatch_margin_weight": mismatch_margin_weight,
            "visual_margin": visual_margin,
            "nonclinical_invariance_weight": nonclinical_invariance_weight,
            "concept_aux_weight": concept_aux_weight,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")

        self.null_margin_weight = float(null_margin_weight)
        self.mismatch_margin_weight = float(mismatch_margin_weight)
        self.visual_margin = float(visual_margin)
        self.nonclinical_invariance_weight = float(nonclinical_invariance_weight)
        self.concept_aux_weight = float(concept_aux_weight)
        self.detach_mismatch_features = bool(detach_mismatch_features)
        self.grounding_mask_fallback = grounding_mask_fallback
        self.log_grounding_diagnostics = bool(log_grounding_diagnostics)
        self.strict_grounding_diagnostics = bool(strict_grounding_diagnostics)
        self._diagnostic_batch_warning_emitted = False
        self.generation_no_repeat_ngram_size = int(generation_no_repeat_ngram_size)
        self.generation_repetition_penalty = float(generation_repetition_penalty)
        self.generation_length_penalty = float(generation_length_penalty)

        self.clinical_supervisor = ClinicalPhraseSupervisor(self.tokenizer)
        hidden_size = int(self.decoder.encoder_decoder.decoder.config.hidden_size)
        self.concept_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(concept_head_dropout),
            nn.Linear(hidden_size, self.clinical_supervisor.num_concepts * 3),
        )

    def configure_optimizers(self):
        grouped_parameters = [
            {"params": self.encoder.parameters(), "lr": self.encoder_lr},
            {"params": self.encoder_projection.parameters(), "lr": self.decoder_lr},
            {"params": self.decoder.parameters(), "lr": self.decoder_lr},
            {"params": self.concept_head.parameters(), "lr": self.decoder_lr},
        ]
        return {"optimizer": torch.optim.AdamW(grouped_parameters, lr=self.decoder_lr)}

    def _encode_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.encoder(images)["last_hidden_state"]
        projected = self.encoder_projection(image_features)["projected_encoder_last_hidden_state"]
        if projected.ndim != 3:
            raise RuntimeError(
                "Expected projected image features with shape [B, visual_tokens, hidden], "
                f"got {tuple(projected.shape)}."
            )
        return projected

    @staticmethod
    def _to_encoder_outputs(features: torch.Tensor):
        return transformers.modeling_outputs.BaseModelOutput(last_hidden_state=features)

    def _decode_visual_tokens(
        self,
        visual_tokens: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.decoder.encoder_decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=self._to_encoder_outputs(visual_tokens),
            return_dict=True,
        )
        return outputs.logits

    @staticmethod
    def _null_visual_tokens(visual_tokens: torch.Tensor) -> torch.Tensor:
        # A fixed zero memory is harder to exploit than a learned null embedding
        # and provides a stable approximation of p(text | no visual evidence).
        return torch.zeros_like(visual_tokens)

    def _clinical_mask(self, label_ids: torch.Tensor) -> torch.Tensor:
        clinical = self.clinical_supervisor.clinical_token_mask(
            label_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        ).to(label_ids.device)
        valid = label_ids.ne(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id is not None:
            valid = valid & label_ids.ne(self.tokenizer.eos_token_id)
        if self.grounding_mask_fallback == "valid":
            no_clinical_phrase = clinical.sum(dim=1).eq(0)
            if no_clinical_phrase.any():
                clinical = clinical.clone()
                clinical[no_clinical_phrase] = valid[no_clinical_phrase]
        return clinical & valid

    def _concept_states(self, reports: Sequence[str], device: torch.device) -> torch.Tensor:
        return self.clinical_supervisor.concept_state_targets(reports, device=device)

    def _mismatch_indices(self, concept_targets: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Select a report-dissimilar in-batch image for every sample.

        Similarity is computed on mentioned concept-state entries. This is safer
        than a simple roll because two adjacent samples can both be normal or can
        contain the same finding. The selected indices need not form a bijection.
        """
        batch_size = concept_targets.shape[0]
        if batch_size < 2:
            return None

        mentioned = concept_targets.ne(self.clinical_supervisor.IGNORE_INDEX)
        positive_or_uncertain = concept_targets.ge(self.clinical_supervisor.POSITIVE) & mentioned
        vectors = positive_or_uncertain.float()
        intersection = vectors @ vectors.t()
        counts = vectors.sum(dim=1, keepdim=True)
        union = counts + counts.t() - intersection
        jaccard = intersection / union.clamp_min(1.0)

        # Reports with no extracted concepts are compared using mention masks.
        empty = counts.squeeze(1).eq(0)
        if empty.any():
            mention_vectors = mentioned.float()
            m_intersection = mention_vectors @ mention_vectors.t()
            m_counts = mention_vectors.sum(dim=1, keepdim=True)
            m_union = m_counts + m_counts.t() - m_intersection
            mention_jaccard = m_intersection / m_union.clamp_min(1.0)
            jaccard[empty] = mention_jaccard[empty]

        jaccard.fill_diagonal_(float("inf"))
        return jaccard.argmin(dim=1)

    def _margin_loss(
        self,
        full_target_logp: torch.Tensor,
        negative_target_logp: torch.Tensor,
        clinical_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gap = full_target_logp - negative_target_logp
        loss = _masked_mean(F.relu(self.visual_margin - gap), clinical_mask)
        mean_gap = _masked_mean(gap.detach(), clinical_mask)
        return loss, mean_gap

    def _concept_auxiliary_loss(
        self,
        visual_tokens: torch.Tensor,
        concept_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = visual_tokens.mean(dim=1)
        logits = self.concept_head(pooled).view(
            visual_tokens.shape[0], self.clinical_supervisor.num_concepts, 3
        )
        flat_targets = concept_targets.reshape(-1)
        valid = flat_targets.ne(self.clinical_supervisor.IGNORE_INDEX)
        if not valid.any():
            zero = logits.sum() * 0.0
            return zero, zero.detach()
        flat_logits = logits.reshape(-1, 3)
        loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=self.clinical_supervisor.IGNORE_INDEX,
        )
        accuracy = (
            flat_logits[valid].argmax(dim=-1).eq(flat_targets[valid]).float().mean().detach()
        )
        return loss, accuracy

    def _compute_grounded_training_loss(self, batch):
        images = batch["encoder_images"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        label_ids = batch["label_ids"]
        reports = batch["labels"]

        visual_tokens = self._encode_visual_tokens(images)
        full_logits = self._decode_visual_tokens(
            visual_tokens, decoder_input_ids, decoder_attention_mask
        )
        ce_loss = F.cross_entropy(
            full_logits.permute(0, 2, 1),
            label_ids,
            ignore_index=self.tokenizer.pad_token_id,
        )
        full_target_logp, valid_mask = _safe_target_log_probs(
            full_logits, label_ids, self.tokenizer.pad_token_id
        )
        clinical_mask = self._clinical_mask(label_ids)
        concept_targets = self._concept_states(reports, device=images.device)

        null_loss = ce_loss.new_zeros(())
        null_gap = ce_loss.new_zeros(())
        invariance_loss = ce_loss.new_zeros(())
        null_logits = None
        if self.null_margin_weight > 0 or self.nonclinical_invariance_weight > 0:
            null_logits = self._decode_visual_tokens(
                self._null_visual_tokens(visual_tokens),
                decoder_input_ids,
                decoder_attention_mask,
            )
            null_target_logp, _ = _safe_target_log_probs(
                null_logits, label_ids, self.tokenizer.pad_token_id
            )
            if self.null_margin_weight > 0:
                null_loss, null_gap = self._margin_loss(
                    full_target_logp, null_target_logp, clinical_mask
                )
            if self.nonclinical_invariance_weight > 0:
                nonclinical_mask = valid_mask & ~clinical_mask
                invariance_loss = _masked_mean(
                    (full_target_logp - null_target_logp).abs(), nonclinical_mask
                )

        mismatch_loss = ce_loss.new_zeros(())
        mismatch_gap = ce_loss.new_zeros(())
        mismatch_indices = None
        if self.mismatch_margin_weight > 0 and images.shape[0] > 1:
            mismatch_indices = self._mismatch_indices(concept_targets)
            if mismatch_indices is not None:
                mismatch_tokens = visual_tokens.index_select(0, mismatch_indices)
                if self.detach_mismatch_features:
                    mismatch_tokens = mismatch_tokens.detach()
                mismatch_logits = self._decode_visual_tokens(
                    mismatch_tokens, decoder_input_ids, decoder_attention_mask
                )
                mismatch_target_logp, _ = _safe_target_log_probs(
                    mismatch_logits, label_ids, self.tokenizer.pad_token_id
                )
                mismatch_loss, mismatch_gap = self._margin_loss(
                    full_target_logp, mismatch_target_logp, clinical_mask
                )

        concept_loss = ce_loss.new_zeros(())
        concept_accuracy = ce_loss.new_zeros(())
        if self.concept_aux_weight > 0:
            concept_loss, concept_accuracy = self._concept_auxiliary_loss(
                visual_tokens, concept_targets
            )

        total_loss = (
            ce_loss
            + self.null_margin_weight * null_loss
            + self.mismatch_margin_weight * mismatch_loss
            + self.nonclinical_invariance_weight * invariance_loss
            + self.concept_aux_weight * concept_loss
        )

        clinical_coverage = clinical_mask.any(dim=1).float().mean().detach()
        metrics = {
            "train_loss": total_loss,
            "train_total_loss": total_loss,
            "train_ce_loss": ce_loss.detach(),
            "train_null_margin_loss": null_loss.detach(),
            "train_mismatch_margin_loss": mismatch_loss.detach(),
            "train_nonclinical_invariance_loss": invariance_loss.detach(),
            "train_concept_aux_loss": concept_loss.detach(),
            "train_null_clinical_gap": null_gap,
            "train_mismatch_clinical_gap": mismatch_gap,
            "train_concept_state_accuracy": concept_accuracy,
            "train_clinical_mask_coverage": clinical_coverage,
        }
        return total_loss, metrics, full_logits

    def training_step(self, batch, batch_idx):
        total_loss, metrics, full_logits = self._compute_grounded_training_loss(batch)
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            batch_size=full_logits.shape[0],
            sync_dist=False,
        )
        return total_loss

    def _diagnostic_warning(self, message: str) -> None:
        """Emit a diagnostic warning once on the global-zero process."""
        if self._diagnostic_batch_warning_emitted:
            return
        self._diagnostic_batch_warning_emitted = True
        if getattr(self, "global_rank", 0) == 0:
            print(f"[visual-grounding diagnostics skipped] {message}")

    def _prepare_diagnostic_teacher_forcing_batch(
        self,
        batch,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Prepare teacher-forcing tensors for val/test grounding diagnostics.

        ``TaskSubset(train=True)`` returns decoder tensors, whereas the original
        validation/test subsets are generation-oriented and commonly return only
        ``encoder_images``, ``labels`` and ``id``.  Therefore diagnostics must not
        directly index ``batch['decoder_input_ids']``.

        The method first reuses decoder tensors when present.  Otherwise it builds
        the same shifted language-model batch from the reference report strings.
        Diagnostics are auxiliary, so malformed/missing validation metadata is
        skipped by default instead of aborting the whole training run.  Set
        ``strict_grounding_diagnostics=True`` to raise the original exception.
        """
        required = (
            "decoder_input_ids",
            "decoder_attention_mask",
            "label_ids",
        )
        if all(key in batch for key in required):
            tensors = tuple(batch[key].to(device) for key in required)
            return tensors  # type: ignore[return-value]

        reports = batch.get("labels")
        if reports is None:
            # Some custom collators use singular ``label``.  Supporting it is
            # harmless and makes this diagnostic independent of that naming detail.
            reports = batch.get("label")

        if reports is None:
            missing = [key for key in required if key not in batch]
            message = (
                "validation/test batch contains neither reference report strings "
                "nor complete teacher-forcing tensors; missing keys: "
                f"{missing}. Available keys: {sorted(batch.keys())}."
            )
            if self.strict_grounding_diagnostics:
                raise KeyError(message)
            self._diagnostic_warning(message)
            return None

        if isinstance(reports, str):
            reports = [reports]
        else:
            reports = [str(report) for report in reports]

        try:
            lm_batch = build_decoder_lm_batch(
                self.tokenizer,
                reports,
                self.decoder_max_len,
                device,
            )
            decoder_input_ids = lm_batch["decoder_input_ids"]
            decoder_attention_mask = lm_batch["decoder_attention_mask"]
            label_ids = lm_batch["label_ids"]
        except Exception as exc:
            message = (
                "failed to reconstruct teacher-forcing tensors from batch reports: "
                f"{type(exc).__name__}: {exc}"
            )
            if self.strict_grounding_diagnostics:
                raise RuntimeError(message) from exc
            self._diagnostic_warning(message)
            return None

        expected_batch = batch["encoder_images"].shape[0]
        if decoder_input_ids.shape[0] != expected_batch:
            message = (
                "reconstructed language-model batch size does not match image batch: "
                f"text={decoder_input_ids.shape[0]}, image={expected_batch}."
            )
            if self.strict_grounding_diagnostics:
                raise RuntimeError(message)
            self._diagnostic_warning(message)
            return None

        return decoder_input_ids, decoder_attention_mask, label_ids

    @torch.no_grad()
    def _grounding_diagnostics(self, batch, prefix: str) -> None:
        if not self.log_grounding_diagnostics:
            return

        images = batch["encoder_images"]
        device = images.device
        diagnostic_batch = self._prepare_diagnostic_teacher_forcing_batch(batch, device)
        if diagnostic_batch is None:
            return
        decoder_input_ids, decoder_attention_mask, label_ids = diagnostic_batch

        visual_tokens = self._encode_visual_tokens(images)
        full_logits = self._decode_visual_tokens(
            visual_tokens,
            decoder_input_ids,
            decoder_attention_mask,
        )
        null_logits = self._decode_visual_tokens(
            self._null_visual_tokens(visual_tokens),
            decoder_input_ids,
            decoder_attention_mask,
        )
        full_logp, _ = _safe_target_log_probs(
            full_logits, label_ids, self.tokenizer.pad_token_id
        )
        null_logp, _ = _safe_target_log_probs(
            null_logits, label_ids, self.tokenizer.pad_token_id
        )
        clinical_mask = self._clinical_mask(label_ids)
        values = {
            f"{prefix}_null_clinical_gap": _masked_mean(full_logp - null_logp, clinical_mask),
        }

        if visual_tokens.shape[0] > 1:
            concept_targets = self._concept_states(
                batch["labels"], device=visual_tokens.device
            )
            indices = self._mismatch_indices(concept_targets)
            if indices is not None:
                mismatch_logits = self._decode_visual_tokens(
                    visual_tokens.index_select(0, indices),
                    decoder_input_ids,
                    decoder_attention_mask,
                )
                mismatch_logp, _ = _safe_target_log_probs(
                    mismatch_logits, label_ids, self.tokenizer.pad_token_id
                )
                values[f"{prefix}_mismatch_clinical_gap"] = _masked_mean(
                    full_logp - mismatch_logp, clinical_mask
                )

        self.log_dict(
            values,
            on_step=False,
            on_epoch=True,
            batch_size=visual_tokens.shape[0],
            sync_dist=False,
        )

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        self._grounding_diagnostics(batch, "val")

    def test_step(self, batch, batch_idx):
        super().test_step(batch, batch_idx)
        self._grounding_diagnostics(batch, "test")

    def generate(
        self,
        num_beams,
        images,
        do_sample: bool = False,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        num_return_sequences: int = 1,
    ):
        """Generation with conservative lexical anti-repetition controls."""
        visual_tokens = self._encode_visual_tokens(images)
        generate_kwargs = {
            "max_length": self.decoder_max_len,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "return_dict_in_generate": True,
            "use_cache": True,
            "encoder_outputs": self._to_encoder_outputs(visual_tokens),
            "no_repeat_ngram_size": self.generation_no_repeat_ngram_size,
            "repetition_penalty": self.generation_repetition_penalty,
            "length_penalty": self.generation_length_penalty,
            "early_stopping": bool(num_beams > 1),
        }
        if top_p is not None:
            generate_kwargs["top_p"] = top_p
        if temperature is not None:
            generate_kwargs["temperature"] = temperature

        outputs = self.decoder.encoder_decoder.generate(**generate_kwargs)
        return outputs["sequences"]
