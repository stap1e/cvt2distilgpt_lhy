"""Form A: rewrite training reports with an offline teacher.

Pipeline per train example: teacher rewrite (current-observable instruction)
-> ``CurrentObservableReportFilterV2`` (identical to the COVAR training filter)
-> new annotation JSON whose TRAIN reports are replaced (val/test untouched).
The training run then points at this file through the config key
``annotation_file``.

Teachers run offline exactly like in ``teacher_score.py`` — never inside
training. ``mock`` is the identity teacher for pipeline smoke tests; use
``file:<path>`` to bridge real rewrites produced elsewhere (recommended for
MedGemma), or ``vlm:<model>`` inside a separate modern-transformers env.

Usage:

    python -m tools.distillation.teacher_rewrite \
        --annotation-in /data/lhy_data/rg/mimic_cxr_chen/annotation.json \
        --annotation-out /data/lhy_data/rg/mimic_cxr_chen/annotation_distilled.json \
        --config config/new_lab/train_covar_distill_sft.yaml \
        --teacher mock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.distillation.current_observable import (
    CurrentObservableReportFilterV2,
)
from tools.distillation.model_loading import load_yaml_config
from tools.distillation.rollout_store import mean, percentile, write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-in", required=True)
    parser.add_argument("--annotation-out", required=True)
    parser.add_argument("--config", required=True, help="YAML config (dataset_dir/ckpt roots).")
    parser.add_argument("--teacher", default="mock", help="'mock' | 'file:<path>' | 'vlm:<model>'")
    parser.add_argument("--device", default="cuda:0", help="Only used by the vlm teacher.")
    parser.add_argument(
        "--max-rewrite",
        type=int,
        default=None,
        help="Only rewrite the first N train examples (teacher-budget guard); "
        "remaining examples keep their filtered original report.",
    )
    parser.add_argument("--report", default=None, help="Sidecar markdown report path.")
    parser.add_argument("--provenance", default=None, help="Per-example provenance JSONL path.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _image_root(config: Dict) -> Path:
    dataset_dir = config.get("dataset_dir")
    if not dataset_dir:
        raise ValueError("Config must define dataset_dir for image-grounded teachers.")
    return Path(dataset_dir).expanduser() / "mimic_cxr_chen" / "mimic_cxr_jpg"


def main() -> None:
    args = parse_args()
    from tools.distillation.teacher import build_teacher

    config = load_yaml_config(args.config)
    image_root = _image_root(config)

    kwargs = {}
    if args.teacher.startswith("vlm:"):
        kwargs["device"] = args.device
    teacher = build_teacher(args.teacher, **kwargs)

    in_path = Path(args.annotation_in).expanduser()
    out_path = Path(args.annotation_out).expanduser()
    report_path = (
        Path(args.report).expanduser()
        if args.report
        else out_path.with_suffix(".report.md")
    )
    provenance_path = (
        Path(args.provenance).expanduser()
        if args.provenance
        else out_path.with_suffix(".provenance.jsonl")
    )

    with in_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    filter_ = CurrentObservableReportFilterV2()
    file_teacher_id_lookup = getattr(teacher, "rows", None)

    stats = {
        "train_examples": len(annotation.get("train", [])),
        "rewritten_by_teacher": 0,
        "kept_original_short_or_empty": 0,
        "sentences_removed": 0,
        "sentences_rewritten": 0,
        "fallback_reports": 0,
        "temporal_before": 0,
        "temporal_after": 0,
    }
    original_lengths: List[float] = []
    final_lengths: List[float] = []

    provenance: List[dict] = []

    for index, example in enumerate(annotation.get("train", [])):
        example_id = str(example.get("id"))
        original = str(example.get("report", ""))
        stats["temporal_before"] += int(filter_.contains_temporal_claim(original))

        use_teacher = (
            args.max_rewrite is None or index < args.max_rewrite
        ) and original.strip()
        teacher_raw = ""
        if use_teacher:
            if isinstance(file_teacher_id_lookup, dict):
                teacher_raw = teacher.rewrite_report(
                    original, image_path=None, key_id=example_id
                )
            else:
                image_path = image_root / str(example.get("image_path", [""])[0])
                teacher_raw = teacher.rewrite_report(
                    original,
                    image_path=str(image_path) if image_path.name else None,
                )

        if use_teacher and teacher_raw.strip() and len(teacher_raw.split()) >= 3:
            stats["rewritten_by_teacher"] += 1
            base_text = teacher_raw
        else:
            if use_teacher:
                stats["kept_original_short_or_empty"] += 1
            base_text = original

        final_text, filter_stats = filter_.filter_report(base_text)
        stats["sentences_removed"] += filter_stats["removed"]
        stats["sentences_rewritten"] += filter_stats["rewritten"]
        stats["fallback_reports"] += filter_stats["fallback"]
        stats["temporal_after"] += int(filter_.contains_temporal_claim(final_text))

        original_lengths.append(float(len(original.split())))
        final_lengths.append(float(len(final_text.split())))

        example["report"] = final_text
        provenance.append(
            {
                "id": example_id,
                "original": original,
                "teacher_raw": teacher_raw,
                "final": final_text,
                "filter": filter_stats,
                "teacher": getattr(teacher, "name", args.teacher),
            }
        )

    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(annotation, handle)
        with provenance_path.open("w", encoding="utf-8") as handle:
            for row in provenance:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fallback_hits = getattr(teacher, "fallback_hits", 0)
    write_markdown_report(
        str(report_path),
        "Teacher rewrite report (Form A)",
        {
            "Run": {
                "teacher": args.teacher,
                "annotation_in": str(in_path),
                "annotation_out": str(out_path),
                "max_rewrite": args.max_rewrite,
                "mock_or_file_fallback_hits": fallback_hits,
            },
            "Counts": stats,
            "Length statistics (words)": {
                "original_mean": mean(original_lengths),
                "final_mean": mean(final_lengths),
                "original_p95": percentile(original_lengths, 0.95),
                "final_p95": percentile(final_lengths, 0.95),
            },
            "Interpretation": [
                "temporal_after should be ~0; a non-zero value means the "
                "filter fallback kept temporal-only reports (expected when "
                "temporal_filter_fallback='keep_original').",
                "final_mean growing a lot versus original_mean means the "
                "teacher expands terse reports; if it shrinks a lot, check "
                "for over-aggressive summarisation in the provenance file.",
            ],
        },
        intro=(
            "mock teacher rewrites are identity — this report then only "
            "measures the filter, which is a valid pipeline smoke test but "
            "NOT a Form A experiment."
            if args.teacher == "mock"
            else None
        ),
    )
    print(f"[teacher_rewrite] wrote annotation to {out_path}")
    print(f"[teacher_rewrite] report at {report_path}")
    print(f"[teacher_rewrite] stats: {stats}")


if __name__ == "__main__":
    main()
