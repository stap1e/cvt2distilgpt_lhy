"""Torch-side dataset for ``train_mode: gkd`` (Form B).

The dataset joins the standard MIMIC TaskSubset examples (images + filtered
reference reports) with the offline-scored rollout pool. Every item carries:

- image tensor (from the base example),
- teacher-forcing tensors of the ROLLOUT text (policy-gradient branch),
- teacher-forcing tensors of the REFERENCE report (COVAR anchor branch),
- per-token teacher scores aligned to the rollout's ``label_ids`` positions
  (text tokens only; EOS/PAD positions are masked out).

Dense-reward alignment contract: ``label_ids = [t_0..t_{n-1}, EOS, PAD..]``
where ``t_i`` are the rollout text tokens, so ``teacher_token_logps[i]``
supervises ``label_ids[i]`` for i < n.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from tools.dataset.dataset import Subset


class GKDRolloutSubset(Dataset):
    """Flat (example, rollout) items for on-policy dense-reward training."""

    def __init__(
        self,
        examples: List[dict],
        selection: Dict[str, List[dict]],
        tokenizer,
        decoder_max_len: int,
        colour_space: str = "RGB",
        transforms=None,
        include_reference_tensors: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.decoder_max_len = decoder_max_len
        self.transforms = transforms
        self.colour_space = colour_space
        self.include_reference_tensors = bool(include_reference_tensors)
        # Subset.tokenize() relies on these attributes.
        self.add_bos_eos_manually = True

        self.examples = examples
        self.flat_items: List[tuple] = []
        missing_ids = 0
        for example in examples:
            rows = selection.get(str(example["id"]))
            if not rows:
                missing_ids += 1
                continue
            for row in rows:
                self.flat_items.append((example, row))
        if missing_ids:
            print(
                f"[GKDRolloutSubset] {missing_ids} training examples have no "
                "scored rollout and are excluded from the GKD dataset."
            )

    def __len__(self) -> int:
        return len(self.flat_items)

    def __getitem__(self, index: int) -> dict:
        example, row = self.flat_items[index]
        image = self.image_loading_and_preprocessing(example["image_file_path"][0])

        rollout_text = str(row["text"])
        rollout_tensors = self.tokenize(rollout_text)

        item = {
            "id": example["id"],
            "encoder_images": image,
            "labels": example["label"],
            "image_filepaths": example["image_file_path"][0],
            "rollout_text": rollout_text,
        }
        item.update({f"rollout_{key}": value for key, value in rollout_tensors.items()})

        if self.include_reference_tensors:
            reference_tensors = self.tokenize(example["label"])
            item.update(
                {f"reference_{key}": value for key, value in reference_tensors.items()}
            )

        teacher_logps = [float(v) for v in row.get("teacher_token_logps", [])]
        # Scores align to label_ids positions 0..n-1 (text tokens only).
        item["gkd_teacher_logps"] = teacher_logps
        item["gkd_teacher_mask"] = [1.0] * len(teacher_logps)
        item["gkd_meta"] = {
            "round": row.get("round", 0),
            "k": row.get("k", 0),
            "source": row.get("source", "sample"),
            "teacher_score": row.get("teacher_score", row.get("teacher_mean_logp", 0.0)),
            "student_mean_logp": row.get("student_mean_logp", 0.0),
        }
        return item

    # Reuse the base Subset mechanics for images and tokenisation.
    def image_loading_and_preprocessing(self, image_file_path):
        from PIL import Image

        image = Image.open(image_file_path)
        image = image.convert(self.colour_space)
        if self.transforms is not None:
            image = self.transforms(image)
        return image

    def tokenize(self, string: str) -> dict:
        string = self.tokenizer.bos_token + string + self.tokenizer.eos_token
        tokenized = self.tokenizer(
            string,
            padding="max_length",
            truncation=True,
            max_length=self.decoder_max_len + 1,
            return_tensors="pt",
        )
        decoder_input_ids = tokenized.input_ids[0]
        decoder_attention_mask = tokenized.attention_mask[0][1:]
        label_ids = decoder_input_ids[1:].detach().clone()
        decoder_input_ids = decoder_input_ids[:-1]
        decoder_input_ids[decoder_input_ids == self.tokenizer.sep_token_id] = (
            self.tokenizer.pad_token_id
        )
        return {
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "label_ids": label_ids,
        }


def gkd_collate(items: List[dict]) -> dict:
    """Collate with manual padding for the variable-length teacher scores."""
    batch: dict = {
        "id": [item["id"] for item in items],
        "labels": [item["labels"] for item in items],
        "image_filepaths": [item["image_filepaths"] for item in items],
        "rollout_text": [item["rollout_text"] for item in items],
        "encoder_images": torch.stack([item["encoder_images"] for item in items], dim=0),
        "gkd_meta": [item["gkd_meta"] for item in items],
    }

    for prefix in ("rollout", "reference"):
        keys = [
            key
            for key in items[0]
            if key.startswith(f"{prefix}_") and key != "rollout_text"
        ]
        for key in keys:
            base = key[len(prefix) + 1 :]
            batch[f"{prefix}_{base}"] = torch.stack(
                [item[key] for item in items], dim=0
            )

    max_scores = max(len(item["gkd_teacher_logps"]) for item in items)
    scores = torch.zeros(len(items), max_scores)
    mask = torch.zeros(len(items), max_scores)
    for row, item in enumerate(items):
        n = len(item["gkd_teacher_logps"])
        if n:
            scores[row, :n] = torch.tensor(item["gkd_teacher_logps"])
            mask[row, :n] = torch.tensor(item["gkd_teacher_mask"])
    batch["gkd_teacher_logps"] = scores
    batch["gkd_teacher_mask"] = mask
    return batch
