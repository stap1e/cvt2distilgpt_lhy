"""JSONL schemas and IO shared by the distillation CLIs.

Schemas
-------
Rollout row (``export_rollouts.py`` output)::

    {"id": str, "idx": int, "split": "train", "round": int,
     "source": "greedy" | "sample", "k": int, "text": str,
     "reference": str, "image_path": str,
     "student_token_ids": [int, ...],
     "student_logp_sum": float, "student_mean_logp": float,
     "num_tokens": int}

Scored rollout row (``teacher_score.py --mode tokens`` output): all rollout
fields plus::

    {"teacher_token_logps": [float, ...],   # aligned to student_token_ids
     "teacher_mean_logp": float,            # mean over aligned scores
     "alignment_coverage": float,
     "teacher": str,                        # e.g. "mock", "file:...", "vlm:..."
     "teacher_score": float}                # ranking score (mode dependent)

Preference pair row (``teacher_score.py --mode rank`` output): compatible with
``tools/preference_rl.load_dpo_pairs_jsonl``::

    {"id": str, "chosen": str, "rejected": str,
     "reward_chosen": float, "reward_rejected": float,
     "error_tags": [str, ...],
     "ref_logp_chosen": float, "ref_logp_rejected": float,  # after fill-ref-logps
     "round": int, "source": {...}, "teacher": str}

Pure Python; no torch imports.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> int:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    in_path = Path(path).expanduser()
    if not in_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {in_path}")
    rows = []
    with in_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {in_path} at line {line_idx}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object in {in_path} at line {line_idx}."
                )
            rows.append(row)
    return rows


def group_by_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(row)
    return dict(grouped)


def write_markdown_report(
    path: str,
    title: str,
    sections: Dict[str, Any],
    intro: Optional[str] = None,
) -> None:
    """Write a compact markdown report of algorithm-relevant statistics.

    ``sections`` values may be scalars, lists of key/values tuples, or nested
    dicts (rendered as bullet lists). Compute statistics (runtimes, GPU
    utilisation, memory) deliberately do not belong here.
    """
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if intro:
        lines.extend([intro, ""])

    def _render_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    for section, content in sections.items():
        lines.append(f"## {section}")
        lines.append("")
        if isinstance(content, dict):
            for key, value in content.items():
                lines.append(f"- {key}: {_render_value(value)}")
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    lines.append(f"- {item[0]}: {_render_value(item[1])}")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append(_render_value(content))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def mean(values: Iterable[float]) -> Optional[float]:
    values = [float(v) for v in values]
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    values = sorted(float(v) for v in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    frac = position - lower
    return values[lower] * (1.0 - frac) + values[upper] * frac
