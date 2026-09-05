"""Scored-rollout pool loading, GKD item selection, and reward math.

Pure Python (no torch) so the selection logic can be unit-tested on a
CPU-only machine; the torch-side dataset wrapper lives in ``gkd_torch.py``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tools.distillation.rollout_store import read_jsonl

_ROUND_RE = re.compile(r"round[_=]?(\d+)")


class GKDPool:
    """Validated view over one or more scored-rollout JSONL files."""

    def __init__(self, rows: List[dict], sources: List[str]) -> None:
        self.rows = rows
        self.sources = sources

    @classmethod
    def load(cls, path: str) -> "GKDPool":
        """Load a scored JSONL file, or the newest-round JSONL in a directory.

        When ``path`` is a directory, the file whose rows carry the largest
        ``round`` value is selected (``gkd_pool_older_rounds`` handling is
        deliberately not implemented; point at an explicit file to mix).
        """
        target = Path(path).expanduser()
        if target.is_dir():
            candidates = sorted(target.glob("*.jsonl"))
            if not candidates:
                raise FileNotFoundError(
                    f"No *.jsonl rollout files found in directory: {target}"
                )
            best_file, best_round = None, None
            for candidate in candidates:
                rows = cls._read_and_validate(candidate)
                rounds = [int(r.get("round", 0)) for r in rows] or [0]
                top_round = max(rounds)
                if best_round is None or top_round > best_round:
                    best_file, best_round = candidate, top_round
            print(
                f"[GKDPool] using {best_file} (round={best_round}) "
                f"from directory {target}"
            )
            target = best_file
        rows = cls._read_and_validate(target)
        return cls(rows, [str(target)])

    @staticmethod
    def _read_and_validate(path: Path) -> List[dict]:
        rows = read_jsonl(str(path))
        for row in rows:
            missing = [
                key
                for key in (
                    "id",
                    "text",
                    "teacher_token_logps",
                    "student_token_ids",
                )
                if key not in row
            ]
            if missing:
                raise ValueError(
                    f"Scored rollout row in {path} is missing fields {missing}. "
                    "Re-run teacher_score.py --mode tokens on the rollout file."
                )
            if len(row["teacher_token_logps"]) != len(row["student_token_ids"]):
                raise ValueError(
                    f"Scored rollout row id={row.get('id')} in {path} has "
                    f"{len(row['teacher_token_logps'])} teacher scores for "
                    f"{len(row['student_token_ids'])} student tokens. The "
                    "alignment is stale; re-run teacher_score.py."
                )
        return rows

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        objective: str = "dense_rl",
        include_greedy: bool = False,
        best_of_k_min_gap: float = 0.0,
    ) -> Dict[str, List[dict]]:
        """Return {example_id: [rollout rows to train on]}.

        - ``dense_rl``: every sampled rollout becomes one training item.
        - ``best_of_k``: the sampled rollout with the highest teacher score
          per example (greedy rows never become SFT targets).
        """
        if objective not in {"dense_rl", "best_of_k"}:
            raise ValueError(
                f"Unsupported gkd_objective: {objective}. "
                "Expected 'dense_rl' or 'best_of_k'."
            )

        grouped: Dict[str, List[dict]] = {}
        for row in self.rows:
            source = str(row.get("source", "sample"))
            is_greedy = source == "greedy"
            if is_greedy and not include_greedy:
                continue
            grouped.setdefault(str(row["id"]), []).append(row)

        selection: Dict[str, List[dict]] = {}
        for example_id, rows in grouped.items():
            if objective == "dense_rl":
                selection[example_id] = rows
                continue

            samples = [row for row in rows if str(row.get("source")) != "greedy"]
            if not samples:
                continue
            scores = [float(row.get("teacher_score", row.get("teacher_mean_logp", 0.0))) for row in samples]
            best_index = max(range(len(samples)), key=lambda i: scores[i])
            if best_of_k_min_gap > 0 and len(scores) > 1:
                gap = scores[best_index] - min(scores)
                if gap < best_of_k_min_gap:
                    continue
            selection[example_id] = [samples[best_index]]
        return selection

    def greedy_teacher_scores(self) -> Dict[str, float]:
        """{example_id: teacher token-logp mean of the greedy rollout}.

        Uses ``teacher_mean_logp`` (the same scale as the per-token dense
        rewards), NOT the ranking scalar ``teacher_score``, so it can serve
        directly as a dense-reward baseline.
        """
        scores = {}
        for row in self.rows:
            if str(row.get("source")) == "greedy" and "teacher_mean_logp" in row:
                scores[str(row["id"])] = float(row["teacher_mean_logp"])
        return scores

    def stats(self) -> dict:
        """Algorithm-relevant pool statistics for the run record."""
        rounds = sorted({int(r.get("round", 0)) for r in self.rows})
        by_source: Dict[str, int] = {}
        for row in self.rows:
            by_source[str(row.get("source", "?"))] = (
                by_source.get(str(row.get("source", "?")), 0) + 1
            )
        coverages = [float(r.get("alignment_coverage", 1.0)) for r in self.rows]
        teacher_means = [
            float(r.get("teacher_mean_logp", 0.0)) for r in self.rows
        ]
        return {
            "num_rows": len(self.rows),
            "num_examples": len({str(r["id"]) for r in self.rows}),
            "rounds": rounds,
            "rows_by_source": by_source,
            "alignment_coverage_mean": (
                sum(coverages) / len(coverages) if coverages else None
            ),
            "teacher_mean_logp_mean": (
                sum(teacher_means) / len(teacher_means) if teacher_means else None
            ),
            "sources": self.sources,
        }


# ----------------------------------------------------------------------
# Dense-reward math (unit-tested helpers)
# ----------------------------------------------------------------------


def whiten(values: Sequence[float]) -> List[float]:
    """Zero-mean unit-variance normalisation of a reward sequence."""
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    if std < 1e-6:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]


def sequence_advantages(
    teacher_logps: Sequence[float],
    baseline: str = "sequence_mean",
    baseline_value: Optional[float] = None,
    clip: float = 3.0,
    normalise: str = "whiten",
) -> List[float]:
    """Per-token advantages for the dense teacher-reward objective.

    - ``baseline='sequence_mean'``: advantage = reward - mean(rewards).
    - ``baseline='greedy_score'``: advantage = reward - greedy_score, where
      ``baseline_value`` is the teacher's sequence score of the greedy rollout.
    ``normalise='whiten'`` rescales advantages per sequence before clipping.
    """
    if baseline not in {"sequence_mean", "greedy_score"}:
        raise ValueError(
            f"Unsupported gkd_advantage_baseline: {baseline}. "
            "Expected 'sequence_mean' or 'greedy_score'."
        )
    if baseline == "greedy_score":
        if baseline_value is None:
            raise ValueError(
                "baseline='greedy_score' requires baseline_value (the greedy "
                "rollout teacher score)."
            )
        centre = float(baseline_value)
    else:
        centre = sum(teacher_logps) / len(teacher_logps) if teacher_logps else 0.0

    advantages = [float(r) - centre for r in teacher_logps]
    if normalise == "whiten":
        advantages = whiten(advantages)
    elif normalise != "none":
        raise ValueError(f"Unsupported gkd_reward_norm: {normalise}")

    if clip and clip > 0:
        advantages = [max(-clip, min(clip, a)) for a in advantages]
    return advantages
