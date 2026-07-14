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


# ============================================================================
# COVAR-V2: Current-Observable Visual Attribution
# ============================================================================

import json as _covar_json
import os as _covar_os
from dataclasses import dataclass as _covar_dataclass

from tools.dataset.mimc_cxr_chen import TaskSubset as _COVARTaskSubset


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


CURRENT_TOKEN_ONLY_PHRASES_V2 = (
    "right",
    "left",
    "bilateral",
    "unilateral",
    "upper lobe",
    "middle lobe",
    "lower lobe",
    "lung base",
    "lung bases",
    "apical",
    "bibasilar",
    "perihilar",
    "retrocardiac",
    "subpleural",
    "small",
    "trace",
    "mild",
    "moderate",
    "large",
    "severe",
)


class CurrentClinicalSupervisorV2:
    """Clinical masks and state labels excluding temporal/change semantics."""

    NEGATIVE = 0
    POSITIVE = 1
    UNCERTAIN = 2
    IGNORE_INDEX = -100

    def __init__(
        self,
        tokenizer,
        report_filter: CurrentObservableReportFilterV2,
        concepts: Mapping[str, Sequence[str]] = DEFAULT_CLINICAL_CONCEPTS,
    ):
        self.tokenizer = tokenizer
        self.report_filter = report_filter
        self.concept_names = tuple(concepts.keys())
        self.aliases = {
            name: tuple(
                dict.fromkeys(
                    alias.lower().strip() for alias in aliases if alias.strip()
                )
            )
            for name, aliases in concepts.items()
        }

        assertion_prefixes = (
            "",
            "no ",
            "without ",
            "no evidence of ",
            "possible ",
            "possibly ",
            "likely ",
            "cannot exclude ",
            "may represent ",
        )
        phrases = list(CURRENT_TOKEN_ONLY_PHRASES_V2)
        for aliases in self.aliases.values():
            for alias in aliases:
                phrases.extend(prefix + alias for prefix in assertion_prefixes)

        token_patterns = []
        for phrase in phrases:
            for text in (phrase, " " + phrase):
                ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
                if ids:
                    token_patterns.append(ids)
        self.token_patterns = tuple(
            sorted(set(token_patterns), key=len, reverse=True)
        )

        self.alias_patterns = {}
        for name, aliases in self.aliases.items():
            patterns = []
            for alias in aliases:
                escaped = re.escape(alias).replace(r"\ ", r"\s+")
                patterns.append(
                    re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
                )
            self.alias_patterns[name] = tuple(patterns)

    @property
    def num_concepts(self) -> int:
        return len(self.concept_names)

    def clinical_token_mask(
        self,
        label_ids: torch.Tensor,
        pad_token_id: int,
        eos_token_id: Optional[int],
    ) -> torch.Tensor:
        if label_ids.ndim != 2:
            raise ValueError(
                f"label_ids must have shape [B, L], got {tuple(label_ids.shape)}"
            )

        mask = torch.zeros_like(label_ids, dtype=torch.bool)
        for batch_index, sequence in enumerate(
            label_ids.detach().cpu().tolist()
        ):
            valid_length = len(sequence)
            for i, token_id in enumerate(sequence):
                if token_id == pad_token_id or (
                    eos_token_id is not None and token_id == eos_token_id
                ):
                    valid_length = i
                    break
            sequence = sequence[:valid_length]

            for pattern in self.token_patterns:
                width = len(pattern)
                if width > valid_length:
                    continue
                for start in range(valid_length - width + 1):
                    if tuple(sequence[start : start + width]) == pattern:
                        mask[batch_index, start : start + width] = True
        return mask

    @staticmethod
    def _mention_state(text: str, match_start: int) -> int:
        prefix = text[max(0, match_start - 120) : match_start]
        if NEGATION_RE.search(prefix):
            return CurrentClinicalSupervisorV2.NEGATIVE
        if UNCERTAINTY_RE.search(prefix):
            return CurrentClinicalSupervisorV2.UNCERTAIN
        return CurrentClinicalSupervisorV2.POSITIVE

    def concept_state_targets(
        self,
        reports: Sequence[str],
        device: torch.device,
        filter_current_only: bool = True,
    ) -> torch.Tensor:
        targets = torch.full(
            (len(reports), self.num_concepts),
            self.IGNORE_INDEX,
            dtype=torch.long,
            device=device,
        )
        priority = {
            self.NEGATIVE: 0,
            self.UNCERTAIN: 1,
            self.POSITIVE: 2,
        }

        for batch_index, raw_report in enumerate(reports):
            report = str(raw_report)
            if filter_current_only:
                report, _ = self.report_filter.filter_report(report)
            report = report.lower()

            for concept_index, concept_name in enumerate(self.concept_names):
                best_state = None
                for pattern in self.alias_patterns[concept_name]:
                    for match in pattern.finditer(report):
                        state = self._mention_state(report, match.start())
                        if (
                            best_state is None
                            or priority[state] > priority[best_state]
                        ):
                            best_state = state
                if best_state is not None:
                    targets[batch_index, concept_index] = best_state
        return targets


@_covar_dataclass
class EvidencePlannerOutputV2:
    memory: torch.Tensor
    concept_state_logits: torch.Tensor
    concept_attention: torch.Tensor


class CurrentEvidencePlannerV2(nn.Module):
    """Compress 576 CvT tokens into generic and concept-specific evidence."""

    def __init__(
        self,
        hidden_size: int,
        num_concepts: int,
        num_generic_queries: int = 16,
        num_heads: int = 8,
        dropout: float = 0.10,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                "decoder hidden size must be divisible by planner_num_heads."
            )

        self.concept_queries = nn.Parameter(
            torch.empty(num_concepts, hidden_size)
        )
        self.generic_queries = nn.Parameter(
            torch.empty(num_generic_queries, hidden_size)
        )
        nn.init.normal_(self.concept_queries, std=0.02)
        nn.init.normal_(self.generic_queries, std=0.02)

        self.concept_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.generic_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.concept_norm1 = nn.LayerNorm(hidden_size)
        self.concept_norm2 = nn.LayerNorm(hidden_size)
        self.generic_norm1 = nn.LayerNorm(hidden_size)
        self.generic_norm2 = nn.LayerNorm(hidden_size)
        self.global_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        ff_size = hidden_size * 2
        self.concept_ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
        )
        self.generic_ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
        )
        self.state_head = nn.Linear(hidden_size, 3)
        self.state_embeddings = nn.Parameter(torch.empty(3, hidden_size))
        nn.init.normal_(self.state_embeddings, std=0.02)

    def forward(self, visual_tokens: torch.Tensor) -> EvidencePlannerOutputV2:
        if visual_tokens.ndim != 3:
            raise ValueError(
                f"visual_tokens must be [B,N,H], got {tuple(visual_tokens.shape)}"
            )
        batch_size = visual_tokens.shape[0]

        concept_queries = self.concept_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        concept_delta, concept_attention = self.concept_attn(
            concept_queries,
            visual_tokens,
            visual_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        concept_tokens = self.concept_norm1(
            concept_queries + self.dropout(concept_delta)
        )
        concept_tokens = self.concept_norm2(
            concept_tokens + self.dropout(self.concept_ffn(concept_tokens))
        )

        state_logits = self.state_head(concept_tokens)
        state_probs = F.softmax(
            state_logits.float(), dim=-1
        ).to(concept_tokens.dtype)
        state_context = torch.einsum(
            "bcs,sh->bch", state_probs, self.state_embeddings
        )
        entropy = -(
            state_probs.float()
            * state_probs.float().clamp_min(1e-8).log()
        ).sum(dim=-1)
        confidence = 1.0 - entropy / math.log(3.0)
        confidence_gate = (
            0.5 + 0.5 * confidence
        ).to(concept_tokens.dtype).unsqueeze(-1)
        concept_memory = (
            self.memory_norm(concept_tokens + state_context) * confidence_gate
        )

        generic_queries = self.generic_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        generic_delta, _ = self.generic_attn(
            generic_queries,
            visual_tokens,
            visual_tokens,
            need_weights=False,
        )
        generic_tokens = self.generic_norm1(
            generic_queries + self.dropout(generic_delta)
        )
        generic_tokens = self.generic_norm2(
            generic_tokens + self.dropout(self.generic_ffn(generic_tokens))
        )

        global_token = self.global_norm(
            visual_tokens.mean(dim=1, keepdim=True)
        )
        memory = torch.cat(
            [global_token, generic_tokens, concept_memory], dim=1
        )
        return EvidencePlannerOutputV2(
            memory=memory,
            concept_state_logits=state_logits,
            concept_attention=concept_attention,
        )


class CvT2DistilGPT2MIMICXRVisualGroundedV2(
    CvT2DistilGPT2MIMICXRChen
):
    """COVAR-V2: current-observable, evidence-bottlenecked report generation."""

    def __init__(
        self,
        *args,
        filter_temporal_train_targets: bool = True,
        drop_temporal_only_train_examples: bool = True,
        temporal_filter_fallback: str = "keep_original",
        planner_num_generic_queries: int = 16,
        planner_num_heads: int = 8,
        planner_dropout: float = 0.10,
        evidence_margin_weight: float = 0.05,
        evidence_margin: float = 0.15,
        concept_aux_weight: float = 0.05,
        planner_localization_weight: float = 0.002,
        grounding_warmup_epochs: int = 2,
        grounding_ramp_epochs: int = 3,
        context_dropout_rate: float = 0.12,
        context_dropout_max_fraction: float = 0.35,
        eos_calibration_weight: float = 0.05,
        eos_margin: float = 0.20,
        null_sample_mean_weight: float = 0.50,
        decoder_base_lr_scale: float = 0.20,
        cross_attention_lr_scale: float = 1.00,
        planner_lr_scale: float = 1.00,
        weight_decay: float = 0.01,
        grounding_mask_fallback: str = "none",
        log_grounding_diagnostics: bool = True,
        strict_grounding_diagnostics: bool = False,
        suppress_temporal_generation: bool = True,
        generation_min_new_tokens: int = 4,
        generation_no_repeat_ngram_size: int = 0,
        generation_repetition_penalty: float = 1.0,
        generation_length_penalty: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.train_mode != "ce":
            raise ValueError(
                "COVAR-V2 currently supports train_mode='ce' only."
            )
        if grounding_mask_fallback not in {"none", "valid"}:
            raise ValueError(
                "grounding_mask_fallback must be 'none' or 'valid'."
            )
        if not 0.0 <= context_dropout_max_fraction < 1.0:
            raise ValueError(
                "context_dropout_max_fraction must be in [0,1)."
            )
        if not 0.0 <= null_sample_mean_weight <= 1.0:
            raise ValueError(
                "null_sample_mean_weight must be in [0,1]."
            )

        self.filter_temporal_train_targets = bool(
            filter_temporal_train_targets
        )
        self.drop_temporal_only_train_examples = bool(
            drop_temporal_only_train_examples
        )
        self.report_filter_v2 = CurrentObservableReportFilterV2(
            fallback=temporal_filter_fallback
        )
        self.current_supervisor_v2 = CurrentClinicalSupervisorV2(
            self.tokenizer, self.report_filter_v2
        )

        hidden_size = int(
            self.decoder.encoder_decoder.decoder.config.hidden_size
        )
        self.evidence_planner_v2 = CurrentEvidencePlannerV2(
            hidden_size=hidden_size,
            num_concepts=self.current_supervisor_v2.num_concepts,
            num_generic_queries=planner_num_generic_queries,
            num_heads=planner_num_heads,
            dropout=planner_dropout,
        )

        self.evidence_margin_weight = float(evidence_margin_weight)
        self.evidence_margin = float(evidence_margin)
        self.concept_aux_weight = float(concept_aux_weight)
        self.planner_localization_weight = float(
            planner_localization_weight
        )
        self.grounding_warmup_epochs = int(grounding_warmup_epochs)
        self.grounding_ramp_epochs = int(grounding_ramp_epochs)
        self.context_dropout_rate = float(context_dropout_rate)
        self.context_dropout_max_fraction = float(
            context_dropout_max_fraction
        )
        self.eos_calibration_weight = float(eos_calibration_weight)
        self.eos_margin = float(eos_margin)
        self.null_sample_mean_weight = float(null_sample_mean_weight)
        self.decoder_base_lr_scale = float(decoder_base_lr_scale)
        self.cross_attention_lr_scale = float(
            cross_attention_lr_scale
        )
        self.planner_lr_scale = float(planner_lr_scale)
        self.weight_decay_v2 = float(weight_decay)
        self.grounding_mask_fallback_v2 = grounding_mask_fallback
        self.log_grounding_diagnostics_v2 = bool(
            log_grounding_diagnostics
        )
        self.strict_grounding_diagnostics_v2 = bool(
            strict_grounding_diagnostics
        )
        self._v2_diagnostic_warning_emitted = False

        self.suppress_temporal_generation = bool(
            suppress_temporal_generation
        )
        self.generation_min_new_tokens = int(
            generation_min_new_tokens
        )
        self.generation_no_repeat_ngram_size_v2 = int(
            generation_no_repeat_ngram_size
        )
        self.generation_repetition_penalty_v2 = float(
            generation_repetition_penalty
        )
        self.generation_length_penalty_v2 = float(
            generation_length_penalty
        )
        self._temporal_bad_words_ids_v2 = (
            self._build_temporal_bad_words_ids_v2()
        )

    # ------------------------------------------------------------------
    # Dataset: only training targets are current-observable distilled.
    # ------------------------------------------------------------------

    def _format_examples_v2(
        self,
        examples: Sequence[dict],
        split: str,
        filter_temporal: bool,
    ):
        removed = rewritten = fallback = dropped = 0
        formatted = []

        for example in examples:
            report = str(example["report"])
            if filter_temporal:
                report, stats = self.report_filter_v2.filter_report(
                    report
                )
                removed += stats["removed"]
                rewritten += stats["rewritten"]
                fallback += stats["fallback"]
                if (
                    stats["fallback"]
                    and self.drop_temporal_only_train_examples
                ):
                    dropped += 1
                    continue

            example["image_file_path"] = example.pop("image_path")
            example["label"] = report
            example["image_file_path"] = [
                _covar_os.path.join(self.dataset_dir, path)
                for path in example["image_file_path"]
            ]
            token_ids = self.chen_tokenizer(example["label"])[
                : self.chen_max_seq_length
            ]
            example["label"] = self.chen_tokenizer.decode(
                token_ids[1:]
            )
            formatted.append(example)

        if filter_temporal:
            print(
                f"[COVAR-V2 COTD/{split}] "
                f"removed_sentences={removed}, "
                f"rewritten_sentences={rewritten}, "
                f"fallback_reports={fallback}, "
                f"dropped_temporal_only={dropped}, "
                f"examples={len(formatted)}"
            )
        return formatted

    def setup(self, stage=None):
        with open(self.labels_file_path) as handle:
            examples = _covar_json.load(handle)

        for split in ("train", "val", "test"):
            images = set()
            for item in examples[split]:
                images.update(item["image_path"])
            print(
                f"{split.capitalize()} set #images: {len(images)}, "
                f"#studies: {len(examples[split])}"
            )

        if stage == "fit" or stage is None:
            self.train_set = _COVARTaskSubset(
                examples=self._format_examples_v2(
                    examples["train"],
                    "train",
                    self.filter_temporal_train_targets,
                ),
                tokenizer=self.tokenizer,
                decoder_max_len=self.decoder_max_len,
                colour_space="RGB",
                transforms=self.train_transforms,
                self_critical=False,
                train=True,
                add_bos_eos_manually=True,
                num_samples=None,
            )
            self.val_set = _COVARTaskSubset(
                examples=self._format_examples_v2(
                    examples["val"], "val", False
                ),
                tokenizer=self.tokenizer,
                decoder_max_len=self.decoder_max_len,
                colour_space="RGB",
                transforms=self.test_transforms,
                add_bos_eos_manually=True,
            )
            print(
                "No. of training & validation examples: "
                f"{len(self.train_set)} & {len(self.val_set)}."
            )

        if stage == "test" or stage is None:
            self.test_set = _COVARTaskSubset(
                examples=self._format_examples_v2(
                    examples["test"], "test", False
                ),
                tokenizer=self.tokenizer,
                decoder_max_len=self.decoder_max_len,
                colour_space="RGB",
                transforms=self.test_transforms,
                add_bos_eos_manually=True,
            )
            print(f"No. of test examples: {len(self.test_set)}.")

    # ------------------------------------------------------------------
    # Differential learning rates preserve the pretrained language prior.
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        decoder_base = []
        decoder_cross = []
        for name, parameter in self.decoder.named_parameters():
            if (
                "crossattention" in name
                or "ln_cross_attn" in name
            ):
                decoder_cross.append(parameter)
            else:
                decoder_base.append(parameter)

        groups = [
            {
                "params": self.encoder.parameters(),
                "lr": self.encoder_lr,
                "weight_decay": self.weight_decay_v2,
            },
            {
                "params": self.encoder_projection.parameters(),
                "lr": self.decoder_lr,
                "weight_decay": self.weight_decay_v2,
            },
            {
                "params": decoder_base,
                "lr": self.decoder_lr * self.decoder_base_lr_scale,
                "weight_decay": self.weight_decay_v2,
            },
            {
                "params": decoder_cross,
                "lr": self.decoder_lr
                * self.cross_attention_lr_scale,
                "weight_decay": self.weight_decay_v2,
            },
            {
                "params": self.evidence_planner_v2.parameters(),
                "lr": self.decoder_lr * self.planner_lr_scale,
                "weight_decay": self.weight_decay_v2,
            },
        ]
        return {
            "optimizer": torch.optim.AdamW(
                groups, lr=self.decoder_lr
            )
        }

    def _encode_visual_tokens_v2(
        self, images: torch.Tensor
    ) -> torch.Tensor:
        image_features = self.encoder(images)["last_hidden_state"]
        projected = self.encoder_projection(image_features)[
            "projected_encoder_last_hidden_state"
        ]
        if projected.ndim != 3:
            raise RuntimeError(
                "Expected projected visual tokens [B,N,H], got "
                f"{tuple(projected.shape)}."
            )
        return projected

    @staticmethod
    def _to_encoder_outputs_v2(memory: torch.Tensor):
        return transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=memory
        )

    def _decode_memory_v2(
        self,
        memory: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.decoder.encoder_decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=self._to_encoder_outputs_v2(memory),
            return_dict=True,
        )
        return outputs.logits

    def encoder_forward(self, images):
        visual_tokens = self._encode_visual_tokens_v2(images)
        plan = self.evidence_planner_v2(visual_tokens)
        return self._to_encoder_outputs_v2(plan.memory)

    def forward(
        self,
        images,
        decoder_input_ids,
        decoder_attention_mask,
    ):
        visual_tokens = self._encode_visual_tokens_v2(images)
        plan = self.evidence_planner_v2(visual_tokens)
        return self._decode_memory_v2(
            plan.memory,
            decoder_input_ids,
            decoder_attention_mask,
        )

    # ------------------------------------------------------------------
    # Scheduled visual grounding and context deprivation.
    # ------------------------------------------------------------------

    def _grounding_scale_v2(self) -> float:
        epoch = int(getattr(self, "current_epoch", 0))
        if epoch < self.grounding_warmup_epochs:
            return 0.0
        if self.grounding_ramp_epochs <= 0:
            return 1.0
        value = (
            epoch - self.grounding_warmup_epochs + 1
        ) / float(self.grounding_ramp_epochs)
        return float(max(0.0, min(1.0, value)))

    def _apply_context_dropout_v2(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scale: float,
    ):
        rate = min(
            self.context_dropout_rate * scale,
            self.context_dropout_max_fraction,
        )
        if not self.training or rate <= 0:
            zero = input_ids.new_zeros((), dtype=torch.float)
            return input_ids, attention_mask, zero

        ids = input_ids.clone()
        mask = attention_mask.clone()
        candidates = mask.bool()
        candidates &= ids.ne(self.tokenizer.bos_token_id)
        candidates &= ids.ne(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id is not None:
            candidates &= ids.ne(self.tokenizer.eos_token_id)

        sampled = torch.rand(ids.shape, device=ids.device) < rate
        dropped = candidates & sampled
        ids[dropped] = self.tokenizer.pad_token_id
        mask[dropped] = 0
        ratio = dropped.float().sum() / candidates.float().sum().clamp_min(
            1.0
        )
        return ids, mask, ratio.detach()

    def _clinical_mask_v2(
        self, label_ids: torch.Tensor
    ) -> torch.Tensor:
        clinical = self.current_supervisor_v2.clinical_token_mask(
            label_ids,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        ).to(label_ids.device)

        valid = label_ids.ne(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id is not None:
            valid &= label_ids.ne(self.tokenizer.eos_token_id)

        if self.grounding_mask_fallback_v2 == "valid":
            empty = clinical.sum(dim=1).eq(0)
            if empty.any():
                clinical = clinical.clone()
                clinical[empty] = valid[empty]
        return clinical & valid

    def _evidence_ablated_visual_tokens_v2(
        self, visual_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Keep feature scale but erase spatially specific patient evidence."""
        detached = visual_tokens.detach()
        sample_mean = detached.mean(
            dim=1, keepdim=True
        ).expand_as(detached)
        batch_mean = detached.mean(
            dim=(0, 1), keepdim=True
        ).expand_as(detached)
        alpha = self.null_sample_mean_weight
        return alpha * sample_mean + (1.0 - alpha) * batch_mean

    def _evidence_margin_loss_v2(
        self,
        full_target_logp: torch.Tensor,
        ablated_target_logp: torch.Tensor,
        clinical_mask: torch.Tensor,
    ):
        gap = full_target_logp - ablated_target_logp.detach()
        loss = _masked_mean(
            F.relu(self.evidence_margin - gap),
            clinical_mask,
        )
        mean_gap = _masked_mean(gap.detach(), clinical_mask)
        return loss, mean_gap

    def _concept_state_loss_v2(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ):
        flat_logits = logits.reshape(-1, 3)
        flat_targets = targets.reshape(-1)
        valid = flat_targets.ne(
            self.current_supervisor_v2.IGNORE_INDEX
        )
        if not valid.any():
            zero = flat_logits.sum() * 0.0
            return zero, zero.detach()

        loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=self.current_supervisor_v2.IGNORE_INDEX,
        )
        accuracy = (
            flat_logits[valid]
            .argmax(dim=-1)
            .eq(flat_targets[valid])
            .float()
            .mean()
            .detach()
        )
        return loss, accuracy

    def _planner_localization_loss_v2(
        self,
        concept_attention: torch.Tensor,
        concept_targets: torch.Tensor,
    ):
        probabilities = concept_attention.float().clamp_min(1e-8)
        entropy = -(
            probabilities * probabilities.log()
        ).sum(dim=-1)
        entropy = entropy / math.log(
            max(2, probabilities.shape[-1])
        )
        active = concept_targets.ne(
            self.current_supervisor_v2.IGNORE_INDEX
        )
        return _masked_mean(entropy, active)

    def _eos_boundary_loss_v2(
        self,
        logits: torch.Tensor,
        label_ids: torch.Tensor,
    ):
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            zero = logits.sum() * 0.0
            return zero, zero.detach(), zero.detach()

        logits32 = logits.float()
        valid = label_ids.ne(self.tokenizer.pad_token_id)
        safe_labels = label_ids.masked_fill(~valid, 0)
        target_logits = logits32.gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        eos_logits = logits32[..., eos_id]

        premature_mask = valid & label_ids.ne(eos_id)
        premature = _masked_mean(
            F.relu(
                eos_logits - target_logits + self.eos_margin
            ),
            premature_mask,
        )

        terminal_mask = valid & label_ids.eq(eos_id)
        other_logits = logits32.clone()
        other_logits[..., eos_id] = torch.finfo(
            other_logits.dtype
        ).min
        max_other = other_logits.max(dim=-1).values
        terminal = _masked_mean(
            F.relu(
                max_other - eos_logits + self.eos_margin
            ),
            terminal_mask,
        )
        return (
            premature + terminal,
            premature.detach(),
            terminal.detach(),
        )

    @staticmethod
    def _capture_rng_state_v2(device: torch.device):
        cpu_state = torch.get_rng_state()
        cuda_state = None
        if device.type == "cuda" and torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state(device)
        return cpu_state, cuda_state

    @staticmethod
    def _restore_rng_state_v2(state, device: torch.device):
        cpu_state, cuda_state = state
        torch.set_rng_state(cpu_state)
        if cuda_state is not None and device.type == "cuda":
            torch.cuda.set_rng_state(cuda_state, device)

    def _compute_training_loss_v2(self, batch):
        images = batch["encoder_images"]
        label_ids = batch["label_ids"]
        reports = batch["labels"]
        scale = self._grounding_scale_v2()

        (
            decoder_input_ids,
            decoder_attention_mask,
            dropout_ratio,
        ) = self._apply_context_dropout_v2(
            batch["decoder_input_ids"],
            batch["decoder_attention_mask"],
            scale,
        )

        visual_tokens = self._encode_visual_tokens_v2(images)
        rng_before = self._capture_rng_state_v2(images.device)
        plan = self.evidence_planner_v2(visual_tokens)
        full_logits = self._decode_memory_v2(
            plan.memory,
            decoder_input_ids,
            decoder_attention_mask,
        )
        rng_after = self._capture_rng_state_v2(images.device)

        ce_loss = F.cross_entropy(
            full_logits.permute(0, 2, 1),
            label_ids,
            ignore_index=self.tokenizer.pad_token_id,
        )
        full_target_logp, _ = _safe_target_log_probs(
            full_logits,
            label_ids,
            self.tokenizer.pad_token_id,
        )
        clinical_mask = self._clinical_mask_v2(label_ids)
        concept_targets = (
            self.current_supervisor_v2.concept_state_targets(
                reports,
                images.device,
                filter_current_only=True,
            )
        )

        evidence_loss = ce_loss.new_zeros(())
        evidence_gap = ce_loss.new_zeros(())
        if scale > 0 and self.evidence_margin_weight > 0:
            # The negative branch is a fixed target. It never changes GPT2.
            with torch.no_grad():
                self._restore_rng_state_v2(
                    rng_before, images.device
                )
                ablated_visual = (
                    self._evidence_ablated_visual_tokens_v2(
                        visual_tokens
                    )
                )
                ablated_plan = self.evidence_planner_v2(
                    ablated_visual
                )
                ablated_logits = self._decode_memory_v2(
                    ablated_plan.memory,
                    decoder_input_ids,
                    decoder_attention_mask,
                )
                ablated_target_logp, _ = (
                    _safe_target_log_probs(
                        ablated_logits,
                        label_ids,
                        self.tokenizer.pad_token_id,
                    )
                )
                self._restore_rng_state_v2(
                    rng_after, images.device
                )

            evidence_loss, evidence_gap = (
                self._evidence_margin_loss_v2(
                    full_target_logp,
                    ablated_target_logp,
                    clinical_mask,
                )
            )

        concept_loss, concept_accuracy = (
            self._concept_state_loss_v2(
                plan.concept_state_logits,
                concept_targets,
            )
        )
        localization_loss = (
            self._planner_localization_loss_v2(
                plan.concept_attention,
                concept_targets,
            )
        )
        (
            eos_loss,
            eos_premature,
            eos_terminal,
        ) = self._eos_boundary_loss_v2(
            full_logits, label_ids
        )

        total_loss = (
            ce_loss
            + scale
            * self.evidence_margin_weight
            * evidence_loss
            + scale
            * self.concept_aux_weight
            * concept_loss
            + scale
            * self.planner_localization_weight
            * localization_loss
            + self.eos_calibration_weight * eos_loss
        )

        metrics = {
            "train_loss": total_loss,
            "train_total_loss": total_loss,
            "train_ce_loss": ce_loss.detach(),
            "train_evidence_margin_loss": (
                evidence_loss.detach()
            ),
            "train_evidence_clinical_gap": evidence_gap,
            "train_concept_aux_loss": concept_loss.detach(),
            "train_concept_state_accuracy": concept_accuracy,
            "train_planner_localization_loss": (
                localization_loss.detach()
            ),
            "train_eos_boundary_loss": eos_loss.detach(),
            "train_eos_premature_loss": eos_premature,
            "train_eos_terminal_loss": eos_terminal,
            "train_context_dropout_ratio": dropout_ratio,
            "train_grounding_scale": torch.tensor(
                scale, device=images.device
            ),
            "train_clinical_mask_coverage": (
                clinical_mask.any(dim=1).float().mean()
            ),
        }
        return total_loss, metrics, full_logits

    def training_step(self, batch, batch_idx):
        total_loss, metrics, logits = (
            self._compute_training_loss_v2(batch)
        )
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            batch_size=logits.shape[0],
            sync_dist=False,
        )
        return total_loss

    # ------------------------------------------------------------------
    # Grounding and generation-collapse diagnostics.
    # ------------------------------------------------------------------

    def _diagnostic_warning_v2(self, message: str):
        if self._v2_diagnostic_warning_emitted:
            return
        self._v2_diagnostic_warning_emitted = True
        if getattr(self, "global_rank", 0) == 0:
            print(f"[COVAR-V2 diagnostics skipped] {message}")

    def _prepare_diagnostic_batch_v2(
        self,
        batch,
        device: torch.device,
    ):
        reports = batch.get("labels", batch.get("label"))
        if reports is None:
            message = (
                "No validation/test reference strings. "
                f"Available keys: {sorted(batch.keys())}"
            )
            if self.strict_grounding_diagnostics_v2:
                raise KeyError(message)
            self._diagnostic_warning_v2(message)
            return None

        if isinstance(reports, str):
            reports = [reports]
        reports = [str(report) for report in reports]
        current_reports = [
            self.report_filter_v2.filter_report(report)[0]
            for report in reports
        ]
        try:
            lm_batch = build_decoder_lm_batch(
                self.tokenizer,
                current_reports,
                self.decoder_max_len,
                device,
            )
        except Exception as exc:
            message = (
                "Could not rebuild diagnostic teacher-forcing "
                f"batch: {type(exc).__name__}: {exc}"
            )
            if self.strict_grounding_diagnostics_v2:
                raise RuntimeError(message) from exc
            self._diagnostic_warning_v2(message)
            return None

        return (
            lm_batch["decoder_input_ids"],
            lm_batch["decoder_attention_mask"],
            lm_batch["label_ids"],
            current_reports,
        )

    @torch.no_grad()
    def _grounding_diagnostics_v2(
        self, batch, prefix: str
    ):
        if not self.log_grounding_diagnostics_v2:
            return

        images = batch["encoder_images"]
        prepared = self._prepare_diagnostic_batch_v2(
            batch, images.device
        )
        if prepared is None:
            return

        (
            decoder_input_ids,
            decoder_attention_mask,
            label_ids,
            current_reports,
        ) = prepared

        visual_tokens = self._encode_visual_tokens_v2(images)
        full_plan = self.evidence_planner_v2(visual_tokens)
        full_logits = self._decode_memory_v2(
            full_plan.memory,
            decoder_input_ids,
            decoder_attention_mask,
        )

        ablated_visual = (
            self._evidence_ablated_visual_tokens_v2(
                visual_tokens
            )
        )
        ablated_plan = self.evidence_planner_v2(
            ablated_visual
        )
        ablated_logits = self._decode_memory_v2(
            ablated_plan.memory,
            decoder_input_ids,
            decoder_attention_mask,
        )

        full_logp, _ = _safe_target_log_probs(
            full_logits,
            label_ids,
            self.tokenizer.pad_token_id,
        )
        ablated_logp, _ = _safe_target_log_probs(
            ablated_logits,
            label_ids,
            self.tokenizer.pad_token_id,
        )
        clinical_mask = self._clinical_mask_v2(label_ids)

        concept_targets = (
            self.current_supervisor_v2.concept_state_targets(
                current_reports,
                images.device,
                filter_current_only=False,
            )
        )
        _, concept_accuracy = (
            self._concept_state_loss_v2(
                full_plan.concept_state_logits,
                concept_targets,
            )
        )
        (
            eos_loss,
            eos_premature,
            eos_terminal,
        ) = self._eos_boundary_loss_v2(
            full_logits, label_ids
        )

        self.log_dict(
            {
                f"{prefix}_evidence_clinical_gap": (
                    _masked_mean(
                        full_logp - ablated_logp,
                        clinical_mask,
                    )
                ),
                f"{prefix}_concept_state_accuracy": (
                    concept_accuracy
                ),
                f"{prefix}_teacher_eos_boundary_loss": (
                    eos_loss
                ),
                f"{prefix}_teacher_eos_premature_loss": (
                    eos_premature
                ),
                f"{prefix}_teacher_eos_terminal_loss": (
                    eos_terminal
                ),
            },
            on_step=False,
            on_epoch=True,
            batch_size=images.shape[0],
            sync_dist=False,
        )

    def _sequence_audit_v2(
        self,
        output_ids: torch.Tensor,
        generated: Sequence[str],
    ):
        batch_size, length = output_ids.shape
        eos_id = self.tokenizer.eos_token_id

        if eos_id is None:
            first_eos = torch.full(
                (batch_size,),
                length,
                device=output_ids.device,
                dtype=torch.long,
            )
        else:
            eos_mask = output_ids.eq(eos_id)
            positions = torch.arange(
                length, device=output_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
            sentinel = torch.full_like(positions, length)
            first_eos = torch.where(
                eos_mask, positions, sentinel
            ).min(dim=1).values

        new_tokens = (first_eos - 1).clamp_min(0).float()
        temporal_rate = torch.tensor(
            sum(
                self.report_filter_v2.contains_temporal_claim(
                    report
                )
                for report in generated
            )
            / max(1, len(generated)),
            device=output_ids.device,
            dtype=torch.float,
        )
        return {
            "mean_generated_tokens": new_tokens.mean(),
            "empty_report_rate": (
                new_tokens.le(0).float().mean()
            ),
            "max_length_hit_rate": (
                first_eos.ge(length - 1).float().mean()
            ),
            "temporal_claim_rate": temporal_rate,
        }

    def validation_step(self, batch, batch_idx):
        output_ids = self.generate(
            1, batch["encoder_images"]
        )
        generated = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        self.val_report_logger.update(
            generated, dicom_ids=batch["id"]
        )
        self.val_chexbert_metrics.update(
            generated,
            batch["labels"],
            ids=batch["id"],
        )
        self.val_coco_metrics.update(
            generated,
            [[text] for text in batch["labels"]],
            ids=batch["id"],
        )

        audit = self._sequence_audit_v2(
            output_ids, generated
        )
        self.log_dict(
            {
                f"val_{key}": value
                for key, value in audit.items()
            },
            on_step=False,
            on_epoch=True,
            batch_size=output_ids.shape[0],
            sync_dist=False,
        )
        self._grounding_diagnostics_v2(batch, "val")

    def test_step(self, batch, batch_idx):
        output_ids = self.generate(
            self.num_test_beams,
            batch["encoder_images"],
        )
        generated = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        self.test_report_logger.update(
            generated, dicom_ids=batch["id"]
        )
        self.test_chexbert_metrics.update(
            generated,
            batch["labels"],
            ids=batch["id"],
        )
        self.test_coco_metrics.update(
            generated,
            [[text] for text in batch["labels"]],
            ids=batch["id"],
        )

        audit = self._sequence_audit_v2(
            output_ids, generated
        )
        self.log_dict(
            {
                f"test_{key}": value
                for key, value in audit.items()
            },
            on_step=False,
            on_epoch=True,
            batch_size=output_ids.shape[0],
            sync_dist=False,
        )
        self._grounding_diagnostics_v2(batch, "test")

    # ------------------------------------------------------------------
    # Explicitly suppress claims unavailable from a current image.
    # ------------------------------------------------------------------

    def _build_temporal_bad_words_ids_v2(self):
        phrases = (
            "previous",
            "prior",
            "compared to",
            "compared with",
            "interval",
            "interval change",
            "unchanged",
            "stable",
            "improved",
            "worsened",
            "new",
            "new since",
            "no relevant change",
            "no significant change",
            "has been removed",
            "has been placed",
            "has been extubated",
            "has been intubated",
        )
        sequences = set()
        for phrase in phrases:
            for text in (phrase, " " + phrase):
                ids = tuple(
                    self.tokenizer.encode(
                        text, add_special_tokens=False
                    )
                )
                if ids:
                    sequences.add(ids)
        return [list(ids) for ids in sorted(sequences)]

    def generate(
        self,
        num_beams,
        images,
        do_sample: bool = False,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        num_return_sequences: int = 1,
    ):
        encoder_outputs = self.encoder_forward(images)
        kwargs = {
            "max_length": self.decoder_max_len,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "forced_eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "return_dict_in_generate": True,
            "use_cache": True,
            "encoder_outputs": encoder_outputs,
            "length_penalty": (
                self.generation_length_penalty_v2
            ),
            "early_stopping": bool(num_beams > 1),
            "renormalize_logits": True,
        }
        if self.generation_min_new_tokens > 0:
            kwargs["min_new_tokens"] = (
                self.generation_min_new_tokens
            )
        if self.generation_no_repeat_ngram_size_v2 > 0:
            kwargs["no_repeat_ngram_size"] = (
                self.generation_no_repeat_ngram_size_v2
            )
        if self.generation_repetition_penalty_v2 != 1.0:
            kwargs["repetition_penalty"] = (
                self.generation_repetition_penalty_v2
            )
        if (
            self.suppress_temporal_generation
            and self._temporal_bad_words_ids_v2
        ):
            kwargs["bad_words_ids"] = (
                self._temporal_bad_words_ids_v2
            )
        if top_p is not None:
            kwargs["top_p"] = top_p
        if temperature is not None:
            kwargs["temperature"] = temperature

        outputs = self.decoder.encoder_decoder.generate(
            **kwargs
        )
        return outputs["sequences"]
