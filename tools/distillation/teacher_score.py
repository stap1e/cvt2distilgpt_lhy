"""Offline teacher scoring/ranking for on-policy distillation.

Three modes:

- ``--mode tokens`` (Form B): per-token teacher log-probs for every rollout,
  aligned onto the student's GPT2 tokenisation (cross-tokenizer alignment
  through character spans). Also scores the reference report, giving the
  teacher-vs-student gap used in the run analysis.
- ``--mode rank`` (Form C): teacher ranking over each example's candidates
  (greedy + K samples, optionally the reference) -> DPO preference pairs in
  the ``load_dpo_pairs_jsonl`` schema.
- ``--mode fill-ref-logps``: compute reference-policy log-probs for
  chosen/rejected texts with a given checkpoint, closing the
  ``dpo_json.py add-ref-logps`` stub so DPO no longer needs the
  reference-free smoke-test mode.

Teachers never run inside training. Run with a different CUDA device (or on
a different machine and ship JSONL back through ``--teacher file:...``).

Examples:

    python -m tools.distillation.teacher_score --mode tokens \
        --rollouts outputs/new_lab/gkd_pool/rollouts_round1.jsonl \
        --out outputs/new_lab/gkd_pool/scored_round1.jsonl \
        --config config/new_lab/train_covar_gkd.yaml --teacher mock

    python -m tools.distillation.teacher_score --mode rank \
        --rollouts outputs/new_lab/gkd_pool/rollouts_round1.jsonl \
        --out outputs/new_lab/dpo/pairs_round1.jsonl \
        --config config/new_lab/train_covar_gkd.yaml --teacher mock \
        --chosen-policy winner --rejected-policy worst_sample

    python -m tools.distillation.teacher_score --mode fill-ref-logps \
        --pairs outputs/new_lab/dpo/pairs_round1.jsonl \
        --out outputs/new_lab/dpo/pairs_round1_ref.jsonl \
        --config config/new_lab/train_covar_dpo_distill.yaml \
        --ckpt-path /path/to/stage1.ckpt --device cuda:5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.distillation.rollout_store import (
    group_by_id,
    mean,
    percentile,
    read_jsonl,
    write_jsonl,
    write_markdown_report,
)
from tools.distillation.token_align import align_teacher_to_student, spans_from_offsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["tokens", "rank", "fill-ref-logps"])

    parser.add_argument("--rollouts", help="[tokens/rank] rollout JSONL from export_rollouts.py.")
    parser.add_argument("--config", required=True, help="Student YAML config.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--report", default=None, help="Sidecar markdown report path.")
    parser.add_argument("--teacher", default="mock", help="'mock' | 'file:<path>' | 'vlm:<model>'")
    parser.add_argument("--device", default="cuda:0", help="Used by vlm teacher and fill-ref-logps.")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    # tokens mode:
    parser.add_argument(
        "--skip-reference-scoring",
        action="store_true",
        help="[tokens] do not score the reference report (faster, loses the gap analysis).",
    )

    # rank mode:
    parser.add_argument(
        "--include-reference",
        type=_str_to_bool,
        default=True,
        help="[rank] include the reference report among ranking candidates.",
    )
    parser.add_argument(
        "--chosen-policy",
        default="winner",
        choices=["winner", "best_sample", "reference"],
        help="[rank] how the chosen text is selected.",
    )
    parser.add_argument(
        "--rejected-policy",
        default="worst_sample",
        choices=["worst_sample", "greedy"],
        help="[rank] how the rejected text is selected.",
    )
    parser.add_argument(
        "--min-score-gap",
        type=float,
        default=0.05,
        help="[rank] skip pairs whose teacher score gap is below this.",
    )

    # fill-ref-logps mode:
    parser.add_argument("--pairs", help="[fill-ref-logps] input preference pairs JSONL.")
    parser.add_argument("--ckpt-path", help="[fill-ref-logps] reference policy checkpoint.")
    parser.add_argument("--batch-size", type=int, default=8, help="[fill-ref-logps] forward batch size.")
    return parser.parse_args()


def _str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def _build_teacher(args):
    from tools.distillation.teacher import build_teacher

    kwargs = {}
    if args.teacher.startswith("vlm:"):
        kwargs["device"] = args.device
    return build_teacher(args.teacher, **kwargs)


# ----------------------------------------------------------------------
# Mode: tokens (Form B)
# ----------------------------------------------------------------------


def run_tokens_mode(args) -> None:
    from tools.distillation.model_loading import load_yaml_config, load_student_tokenizer

    config = load_yaml_config(args.config)
    student_tokenizer = load_student_tokenizer(config)
    teacher = _build_teacher(args)

    rows = read_jsonl(args.rollouts)
    out_rows: List[dict] = []
    coverages, sample_means, greedy_means, ref_means, gaps = [], [], [], [], []

    for row in rows:
        text = str(row["text"])
        encoded = student_tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        student_ids = list(encoded["input_ids"])
        student_spans = spans_from_offsets(encoded["offset_mapping"])

        teacher_out = teacher.score_tokens(
            text,
            reference=row.get("reference"),
            image_path=row.get("image_path"),
        )
        teacher_spans = spans_from_offsets(teacher_out["teacher_token_offsets"])
        scores, coverage = align_teacher_to_student(
            student_spans, teacher_spans, teacher_out["teacher_token_logps"]
        )
        if len(scores) != len(student_ids):
            # Degenerate tokenisations (e.g. empty text): fall back to zeros.
            scores = [0.0] * len(student_ids)
            coverage = 0.0

        rank_score = teacher.rank_score(
            text, reference=row.get("reference"), image_path=row.get("image_path")
        )

        teacher_mean = mean(scores) or 0.0
        out_row = dict(row)
        out_row.update(
            {
                "teacher_token_logps": scores,
                "teacher_mean_logp": float(teacher_mean),
                "alignment_coverage": float(coverage),
                "teacher": getattr(teacher, "name", args.teacher),
                "teacher_score": float(rank_score),
            }
        )

        if not args.skip_reference_scoring and row.get("reference"):
            ref_out = teacher.score_tokens(
                str(row["reference"]),
                reference=row.get("reference"),
                image_path=row.get("image_path"),
            )
            ref_mean = mean(ref_out["teacher_token_logps"]) or 0.0
            out_row["teacher_reference_mean_logp"] = float(ref_mean)
            ref_means.append(float(ref_mean))
            gaps.append(float(teacher_mean) - float(ref_mean))

        out_rows.append(out_row)
        coverages.append(float(coverage))
        if row.get("source") == "greedy":
            greedy_means.append(float(teacher_mean))
        else:
            sample_means.append(float(teacher_mean))

        if args.max_items is not None and len(out_rows) >= args.max_items:
            break

    report_path = args.report or str(Path(args.out).with_suffix(".report.md"))
    fallback_hits = getattr(teacher, "fallback_hits", 0)
    if not args.dry_run:
        write_jsonl(out_rows, args.out)
    _tokens_report(args, out_rows, coverages, sample_means, greedy_means, ref_means, gaps, fallback_hits, report_path)
    print(f"[teacher_score/tokens] wrote {len(out_rows)} rows to {args.out}")


def _tokens_report(args, rows, coverages, sample_means, greedy_means, ref_means, gaps, fallback_hits, report_path):
    write_markdown_report(
        report_path,
        "Teacher token-scoring report (Form B)",
        {
            "Run": {
                "teacher": args.teacher,
                "rollouts": args.rollouts,
                "rows_scored": len(rows),
                "mock_or_file_fallback_hits": fallback_hits,
            },
            "Alignment": {
                "coverage_mean": mean(coverages),
                "coverage_p05": percentile(coverages, 0.05),
            },
            "Teacher log-prob of student tokens": {
                "sample_rollouts_mean": mean(sample_means),
                "greedy_rollouts_mean": mean(greedy_means),
            },
            "Teacher log-prob of reference tokens": {
                "reference_mean": mean(ref_means),
                "gap_rollout_minus_reference_mean": mean(gaps),
                "gap_p95": percentile(gaps, 0.95) if gaps else None,
            },
            "Interpretation": [
                "gap_rollout_minus_reference_mean < 0 means the teacher still "
                "prefers reference reports to student rollouts; as GKD rounds "
                "progress this gap should shrink toward 0.",
                "coverage_mean below ~0.9 signals tokenizer-alignment noise "
                "(many nearest-neighbour fallbacks) — inspect examples before "
                "trusting dense rewards.",
            ],
        },
        intro=(
            "mock teacher rows are pipeline smoke tests only — do not report "
            "them as experimental results."
            if args.teacher == "mock"
            else None
        ),
    )


# ----------------------------------------------------------------------
# Mode: rank (Form C)
# ----------------------------------------------------------------------


def run_rank_mode(args) -> None:
    teacher = _build_teacher(args)
    rows = read_jsonl(args.rollouts)
    grouped = group_by_id(rows)

    pairs: List[dict] = []
    stats = {
        "examples": len(grouped),
        "pairs_written": 0,
        "skipped_identical": 0,
        "skipped_gap": 0,
        "skipped_no_candidates": 0,
        "best_sample_beats_greedy": 0,
        "best_sample_beats_reference": 0,
        "reference_wins_overall": 0,
    }
    gaps = []

    for example_id, example_rows in grouped.items():
        samples = [r for r in example_rows if r.get("source") != "greedy"]
        greedy_rows = [r for r in example_rows if r.get("source") == "greedy"]
        if not samples:
            stats["skipped_no_candidates"] += 1
            continue
        reference = str(samples[0].get("reference", ""))
        image_path = samples[0].get("image_path")

        def score_of(text: str) -> float:
            return float(
                teacher.rank_score(text, reference=reference, image_path=image_path)
            )

        sample_scores = [(r, score_of(str(r["text"]))) for r in samples]
        best_row, best_score = max(sample_scores, key=lambda item: item[1])
        worst_row, worst_score = min(sample_scores, key=lambda item: item[1])
        greedy_row = greedy_rows[0] if greedy_rows else None
        greedy_score = score_of(str(greedy_row["text"])) if greedy_row else None

        reference_score = None
        if reference and args.include_reference:
            reference_score = score_of(reference)

        if greedy_score is not None:
            stats["best_sample_beats_greedy"] += int(best_score > greedy_score)
        if reference_score is not None:
            stats["best_sample_beats_reference"] += int(best_score > reference_score)

        if args.chosen_policy == "reference":
            if reference_score is None:
                stats["skipped_no_candidates"] += 1
                continue
            chosen_text, chosen_score, chosen_source = reference, reference_score, "reference"
        elif args.chosen_policy == "best_sample":
            chosen_text, chosen_score, chosen_source = str(best_row["text"]), best_score, "best_sample"
        else:  # winner
            if reference_score is not None and reference_score >= best_score:
                chosen_text, chosen_score, chosen_source = reference, reference_score, "reference"
                stats["reference_wins_overall"] += 1
            else:
                chosen_text, chosen_score, chosen_source = str(best_row["text"]), best_score, "best_sample"

        if args.rejected_policy == "greedy":
            if greedy_row is None:
                rejected_text, rejected_score, rejected_source = str(worst_row["text"]), worst_score, "worst_sample"
            else:
                rejected_text, rejected_score, rejected_source = str(greedy_row["text"]), greedy_score, "greedy"
        else:
            rejected_text, rejected_score, rejected_source = str(worst_row["text"]), worst_score, "worst_sample"

        if not chosen_text.strip() or not rejected_text.strip():
            stats["skipped_no_candidates"] += 1
            continue
        if chosen_text == rejected_text:
            stats["skipped_identical"] += 1
            continue
        gap = chosen_score - rejected_score
        if gap < args.min_score_gap:
            stats["skipped_gap"] += 1
            continue
        gaps.append(gap)

        pairs.append(
            {
                "id": example_id,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "reward_chosen": float(chosen_score),
                "reward_rejected": float(rejected_score),
                "error_tags": _error_tags(teacher, rejected_text, reference),
                "round": int(samples[0].get("round", 0)),
                "teacher": getattr(teacher, "name", args.teacher),
                "source": {"chosen": chosen_source, "rejected": rejected_source},
            }
        )
        stats["pairs_written"] += 1

    report_path = args.report or str(Path(args.out).with_suffix(".report.md"))
    if not args.dry_run:
        write_jsonl(pairs, args.out)
    _rank_report(args, pairs, stats, gaps, report_path)
    print(f"[teacher_score/rank] wrote {len(pairs)} pairs to {args.out}")


def _error_tags(teacher, rejected_text: str, reference: str) -> List[str]:
    if hasattr(teacher, "reward_evaluator"):
        score = teacher.reward_evaluator.score_one(rejected_text, reference)
        return [name for name, value in score.get("penalties", {}).items() if value > 0]
    return []


def _rank_report(args, pairs, stats, gaps, report_path):
    examples = max(1, stats["examples"])
    write_markdown_report(
        report_path,
        "Teacher preference-ranking report (Form C)",
        {
            "Run": {
                "teacher": args.teacher,
                "rollouts": args.rollouts,
                "chosen_policy": args.chosen_policy,
                "rejected_policy": args.rejected_policy,
                "min_score_gap": args.min_score_gap,
                "include_reference": args.include_reference,
            },
            "Counts": stats,
            "Teacher-student gap rates": {
                "best_sample_beats_greedy_rate": stats["best_sample_beats_greedy"] / examples,
                "best_sample_beats_reference_rate": stats["best_sample_beats_reference"] / examples,
                "reference_wins_overall_rate": stats["reference_wins_overall"] / examples,
            },
            "Chosen-rejected teacher score gap": {
                "mean": mean(gaps),
                "p05": percentile(gaps, 0.05),
                "p95": percentile(gaps, 0.95),
            },
            "Interpretation": [
                "best_sample_beats_reference_rate > 0.5 means the teacher "
                "already prefers some on-policy rollouts to the (filtered) "
                "reference — evidence that on-policy preference data carries "
                "signal beyond imitation.",
                "A large skipped_gap fraction means the teacher is nearly "
                "indifferent between candidates: raise K or lower "
                "--min-score-gap to keep pairs.",
            ],
        },
        intro=(
            "mock teacher rows are pipeline smoke tests only."
            if args.teacher == "mock"
            else None
        ),
    )


# ----------------------------------------------------------------------
# Mode: fill-ref-logps (Form C support)
# ----------------------------------------------------------------------


def run_fill_ref_logps_mode(args) -> None:
    import torch
    from PIL import Image

    from tools.distillation.model_loading import load_model_from_config
    from tools.preference_rl import (
        build_decoder_lm_batch,
        sequence_logprobs_from_logits,
    )

    device = torch.device(args.device)
    model = load_model_from_config(args.config, args.ckpt_path, device)
    model.setup("fit")

    image_by_id = {}
    for example in model.train_set.examples:
        image_by_id[str(example["id"])] = example["image_file_path"][0]

    pairs = read_jsonl(args.pairs)
    out_rows: List[dict] = []
    skipped_missing_image = 0
    filled = 0

    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_size):
            chunk = [p for p in pairs[start : start + args.batch_size] if str(p["id"]) in image_by_id]
            skipped_missing_image += sum(
                1
                for p in pairs[start : start + args.batch_size]
                if str(p["id"]) not in image_by_id
            )
            if not chunk:
                continue
            images = torch.stack(
                [
                    model.test_transforms(
                        Image.open(image_by_id[str(p["id"])]).convert("RGB")
                    )
                    for p in chunk
                ],
                dim=0,
            ).to(device)

            for field, out_field in (("chosen", "ref_logp_chosen"), ("rejected", "ref_logp_rejected")):
                texts = [str(p[field]) for p in chunk]
                lm = build_decoder_lm_batch(
                    model.tokenizer, texts, model.decoder_max_len, device
                )
                logits = model(
                    images, lm["decoder_input_ids"], lm["decoder_attention_mask"]
                )
                logps = sequence_logprobs_from_logits(
                    logits, lm["label_ids"], model.tokenizer.pad_token_id
                )
                for pair, value in zip(chunk, logps.tolist()):
                    pair[out_field] = float(value)

            for pair in chunk:
                out_rows.append(pair)
                filled += 1

    report_path = args.report or str(Path(args.out).with_suffix(".report.md"))
    if not args.dry_run:
        write_jsonl(out_rows, args.out)
    write_markdown_report(
        report_path,
        "Reference log-prob fill report (Form C)",
        {
            "Run": {
                "config": args.config,
                "reference_ckpt": args.ckpt_path,
                "pairs_in": len(pairs),
                "pairs_filled": filled,
                "skipped_missing_image": skipped_missing_image,
            },
            "Reference log-probs": {
                "chosen_mean": mean([p.get("ref_logp_chosen", 0.0) for p in out_rows]),
                "rejected_mean": mean([p.get("ref_logp_rejected", 0.0) for p in out_rows]),
            },
        },
    )
    print(f"[teacher_score/fill-ref-logps] wrote {len(out_rows)} pairs to {args.out}")


def main() -> None:
    args = parse_args()
    if args.mode == "tokens":
        run_tokens_mode(args)
    elif args.mode == "rank":
        run_rank_mode(args)
    else:
        run_fill_ref_logps_mode(args)


if __name__ == "__main__":
    main()
