"""Export on-policy rollouts (greedy + K samples) from a student checkpoint.

First step of both Form B (GKD) and Form C (teacher-ranked DPO). Run this as
a SEPARATE job from training — it is offline tooling and never runs
concurrently with a training process on the same GPU.

Usage (from the repository root):

    python -m tools.distillation.export_rollouts \
        --config config/new_lab/train_covar_gkd.yaml \
        --ckpt-path /path/to/epoch=X-val_ce_f1_macro=Y.ckpt \
        --split train --num-samples 4 --round 1 \
        --out outputs/new_lab/gkd_pool/rollouts_round1.jsonl \
        --device cuda:5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.distillation.current_observable import (
    CurrentObservableReportFilterV2,
)
from tools.distillation.model_loading import load_model_from_config, load_yaml_config
from tools.distillation.rollout_store import (
    mean,
    percentile,
    write_jsonl,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config of the student.")
    parser.add_argument("--ckpt-path", required=True, help="Student checkpoint.")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--num-samples", type=int, default=4, help="K sampled rollouts per example.")
    parser.add_argument("--no-greedy", action="store_true", help="Skip the greedy rollout.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--round", type=int, default=0, help="On-policy round index.")
    parser.add_argument("--out", required=True, help="Output rollout JSONL path.")
    parser.add_argument("--report", default=None, help="Sidecar markdown report path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _student_logps(model, images_per_candidate, texts, device, chunk: int = 48):
    """Teacher-forced logp_sum / token count for each candidate text."""
    from tools.preference_rl import (
        build_decoder_lm_batch,
        sequence_logprobs_from_logits,
    )

    logp_sums, token_counts = [], []
    for start in range(0, len(texts), chunk):
        chunk_texts = texts[start : start + chunk]
        chunk_images = images_per_candidate[start : start + chunk].to(device)
        lm = build_decoder_lm_batch(
            model.tokenizer, chunk_texts, model.decoder_max_len, device
        )
        logits = model(
            chunk_images,
            lm["decoder_input_ids"],
            lm["decoder_attention_mask"],
        )
        sums = sequence_logprobs_from_logits(
            logits, lm["label_ids"], model.tokenizer.pad_token_id
        )
        counts = lm["label_ids"].ne(model.tokenizer.pad_token_id).sum(dim=-1)
        logp_sums.extend(float(v) for v in sums.tolist())
        token_counts.extend(int(v) for v in counts.tolist())
    return logp_sums, token_counts


def main() -> None:
    args = parse_args()
    import torch

    device = torch.device(args.device)
    config = load_yaml_config(args.config)
    if args.batch_size is not None:
        config["mbatch_size"] = args.batch_size

    model = load_model_from_config(args.config, args.ckpt_path, device)
    if args.batch_size is not None:
        model.mbatch_size = args.batch_size
    if int(getattr(model, "num_workers", 0) or 0) <= 0:
        model.num_workers = 1

    model.setup("fit" if args.split == "train" else "test")
    dataloader = (
        model.train_dataloader(shuffle=False)
        if args.split == "train"
        else model.val_dataloader()
    )

    out_path = Path(args.out).expanduser()
    report_path = (
        Path(args.report).expanduser()
        if args.report
        else out_path.with_suffix(".report.md")
    )

    print(
        f"[rollouts] split={args.split} num_samples={args.num_samples} "
        f"round={args.round}"
    )
    print(f"[rollouts] out={out_path}")

    rows = []
    num_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["encoder_images"].to(device)
            ids = [str(i) for i in batch["id"]]
            references = [str(r) for r in batch["labels"]]
            image_paths = [
                str(p) for p in batch.get("image_filepaths", [""] * len(ids))
            ]
            batch_size = images.shape[0]

            greedy_texts = None
            if not args.no_greedy:
                greedy_ids = model.generate(1, images)
                greedy_texts = model.tokenizer.batch_decode(
                    greedy_ids, skip_special_tokens=True
                )

            sampled_texts = None
            if args.num_samples > 0:
                sampled_ids = model.generate(
                    1,
                    images,
                    do_sample=True,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    num_return_sequences=args.num_samples,
                )
                # HF returns [B * K, L] with K consecutive sequences per example.
                decoded = model.tokenizer.batch_decode(
                    sampled_ids, skip_special_tokens=True
                )
                sampled_texts = [
                    decoded[i * args.num_samples : (i + 1) * args.num_samples]
                    for i in range(batch_size)
                ]

            candidates_per_example = []
            for i in range(batch_size):
                candidates = []
                if greedy_texts is not None:
                    candidates.append(("greedy", 0, greedy_texts[i]))
                if sampled_texts is not None:
                    for k, text in enumerate(sampled_texts[i]):
                        candidates.append(("sample", k + 1, text))
                candidates_per_example.append(candidates)

            flat_texts, flat_images, flat_meta = [], [], []
            for i, candidates in enumerate(candidates_per_example):
                for source, k, text in candidates:
                    flat_meta.append((i, source, k))
                    flat_texts.append(text)
                    flat_images.append(images[i])

            flat_images = torch.stack(flat_images, dim=0)
            logp_sums, token_counts = _student_logps(
                model, flat_images, flat_texts, device
            )

            for (i, source, k), text, logp_sum, n_tokens in zip(
                flat_meta, flat_texts, logp_sums, token_counts
            ):
                token_ids = model.tokenizer.encode(text, add_special_tokens=False)
                rows.append(
                    {
                        "id": ids[i],
                        "idx": num_examples + i,
                        "split": args.split,
                        "round": args.round,
                        "source": source,
                        "k": k,
                        "text": text,
                        "reference": references[i],
                        "image_path": image_paths[i],
                        "student_token_ids": token_ids,
                        "student_logp_sum": float(logp_sum),
                        "student_mean_logp": float(logp_sum) / max(1, n_tokens),
                        "num_tokens": int(n_tokens),
                    }
                )

            num_examples += batch_size
            print(f"[rollouts] examples={num_examples} rows={len(rows)}", end="\r")
            if args.max_items is not None and num_examples >= args.max_items:
                break

    if not args.dry_run:
        written = write_jsonl(rows, str(out_path))
        _write_report(args, rows, str(report_path), written, num_examples)
        print(f"\n[rollouts] wrote {written} rows to {out_path}")
        print(f"[rollouts] report at {report_path}")
    else:
        print("\n[rollouts] dry-run: nothing written")


def _write_report(args, rows, report_path, written, num_examples):
    filter_ = CurrentObservableReportFilterV2()
    samples = [r for r in rows if r["source"] == "sample"]
    greedy_by_id = {r["id"]: r["text"] for r in rows if r["source"] == "greedy"}
    dup_with_greedy = sum(
        1
        for example_id in {r["id"] for r in samples}
        if example_id in greedy_by_id
        and greedy_by_id[example_id] in [s["text"] for s in samples if s["id"] == example_id]
    )
    lengths = [r["num_tokens"] for r in rows]
    temporal = [filter_.contains_temporal_claim(r["text"]) for r in rows]

    write_markdown_report(
        report_path,
        "Rollout export report",
        {
            "Export": {
                "config": args.config,
                "ckpt_path": args.ckpt_path,
                "split": args.split,
                "round": args.round,
                "num_samples_per_example": args.num_samples,
                "greedy_included": not args.no_greedy,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            "Counts": {
                "examples": num_examples,
                "rows_written": written,
                "sample_rows": len(samples),
                "greedy_rows": len(rows) - len(samples),
            },
            "Rollout statistics": {
                "mean_tokens": mean(lengths),
                "p05_tokens": percentile(lengths, 0.05),
                "p95_tokens": percentile(lengths, 0.95),
                "temporal_claim_rate": (
                    sum(temporal) / len(temporal) if temporal else None
                ),
                "examples_with_greedy_duplicate_sample": dup_with_greedy,
                "mean_student_logp_sum": mean([r["student_logp_sum"] for r in rows]),
            },
        },
    )


if __name__ == "__main__":
    main()
