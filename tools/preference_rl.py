import json
import os
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


_OPTIONAL_DPO_FIELDS = (
    'ref_logp_chosen',
    'ref_logp_rejected',
    'reward_chosen',
    'reward_rejected',
    'error_tags',
)


def load_dpo_pairs_jsonl(path: Optional[str]) -> Dict[str, dict]:
    """
    Load DPO preference pairs from a JSONL file.

    Each line must contain an ``id`` and may contain ``chosen``, ``rejected``,
    precomputed reference log-probabilities, reward values, and error tags.
    IDs are normalized to strings so they match batched dataset IDs robustly.
    """
    if path is None:
        return {}

    if not os.path.isfile(path):
        raise FileNotFoundError(f'DPO pair JSONL file not found: {path}')

    pairs = {}
    with open(path) as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            if 'id' not in row:
                raise ValueError(f'Missing "id" in DPO pair file {path} at line {line_idx}.')

            pair = {}
            if 'chosen' in row:
                pair['chosen'] = row['chosen']
            if 'rejected' in row:
                pair['rejected'] = row['rejected']
            for field in _OPTIONAL_DPO_FIELDS:
                if field in row:
                    pair[field] = row[field]

            pairs[str(row['id'])] = pair

    return pairs


def build_decoder_lm_batch(tokenizer, texts: Iterable[str], max_len: int, device: torch.device) -> dict:
    """
    Build decoder teacher-forcing tensors using the same shift convention as
    ``tools/dataset/dataset.py::Subset.tokenize``.
    """
    strings = [tokenizer.bos_token + text + tokenizer.eos_token for text in texts]
    tokenized = tokenizer(
        strings,
        padding='max_length',
        truncation=True,
        max_length=max_len + 1,
        return_tensors='pt',
    )

    decoder_input_ids = tokenized.input_ids[:, :-1]
    decoder_attention_mask = tokenized.attention_mask[:, 1:]
    label_ids = tokenized.input_ids[:, 1:].detach().clone()

    if tokenizer.sep_token_id is not None:
        decoder_input_ids = decoder_input_ids.clone()
        decoder_input_ids[decoder_input_ids == tokenizer.sep_token_id] = tokenizer.pad_token_id

    return {
        'decoder_input_ids': decoder_input_ids.to(device),
        'decoder_attention_mask': decoder_attention_mask.to(device),
        'label_ids': label_ids.to(device),
    }


def sequence_logprobs_from_logits(
        logits: torch.Tensor,
        label_ids: torch.Tensor,
        pad_token_id: int,
        normalize: bool = False,
) -> torch.Tensor:
    """
    Compute summed sequence log-probabilities from token logits and labels.
    """
    safe_label_ids = label_ids.masked_fill(label_ids == pad_token_id, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=safe_label_ids.unsqueeze(-1)).squeeze(-1)

    mask = label_ids.ne(pad_token_id)
    sequence_log_probs = (token_log_probs * mask).sum(dim=-1)

    if normalize:
        lengths = mask.sum(dim=-1).clamp_min(1)
        sequence_log_probs = sequence_log_probs / lengths

    return sequence_log_probs


def compute_dpo_loss(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        beta: float,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the standard DPO loss and lightweight detached metrics.
    """
    policy_margin = policy_chosen_logps - policy_rejected_logps
    ref_margin = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_margin - ref_margin)
    loss = -F.logsigmoid(logits).mean()

    metrics = {
        'dpo_margin': (policy_margin - ref_margin).detach().mean(),
        'dpo_chosen_logp_mean': policy_chosen_logps.detach().mean(),
        'dpo_rejected_logp_mean': policy_rejected_logps.detach().mean(),
    }
    return loss, metrics
