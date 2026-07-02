"""Utilities for building DPO JSONL data for CvT2DistilGPT2 RRG.

Provides split-aware CE prediction generation and raw DPO pair building.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_VALID_SPLITS = {"train", "val", "test"}
_KNOWN_TEST_SIZE = 2210
_FULL_MIMIC_TRAIN_SIZE = 145471
_ID_FIELDS = ["id", "sample_id", "study_id", "dicom_id", "uid"]
_REFERENCE_FIELDS = ["reference", "label", "labels", "target", "gt", "ground_truth", "chosen"]
_GENERATED_FIELDS = ["generated", "prediction", "pred", "generated_report", "rejected"]


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file, skipping empty lines and reporting line-numbered JSON errors."""
    jsonl_path = Path(path).expanduser()
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {jsonl_path} at line {line_idx}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {jsonl_path} at line {line_idx}, got {type(row).__name__}.")
            rows.append(row)
    return rows


def _config_kwargs_for_model(config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """Prepare config kwargs for the MIMIC LightningModule constructor."""
    config_kwargs = dict(config)
    config_file = Path(config_path).expanduser()

    config_kwargs.setdefault("warm_start_modules", False)
    config_kwargs["warm_start_modules"] = False

    if "exp_dir_trial" not in config_kwargs:
        config_kwargs["exp_dir_trial"] = config_kwargs.get("exp_dir", "./outputs")

    if int(config_kwargs.get("num_workers", 0) or 0) <= 0:
        print(
            "[Info] num_workers <= 0 would be incompatible with the module dataloader "
            "prefetch_factor; using num_workers=1 for generate-preds."
        )
        config_kwargs["num_workers"] = 1

    config_kwargs["train_mode"] = "ce"
    config_kwargs["dpo_pair_path"] = None
    if "warm_start_ckpt_path" in config_kwargs:
        config_kwargs["warm_start_ckpt_path"] = None

    # Keep this only as metadata in config files; constructor accepts **kwargs, but
    # forcing the concrete class avoids depending on dlhpcstarter importer state.
    config_kwargs.pop("module", None)
    config_kwargs.pop("definition", None)
    config_kwargs.setdefault("_config_path", str(config_file))
    return config_kwargs


def load_model_from_config(
        config_path: str,
        ckpt_path: str,
        device: torch.device,
        strict_load: bool = False,
) -> Any:
    """Load the MIMIC CvT2DistilGPT2 model from YAML config and checkpoint."""
    from cvt2distilgpt2_mimic_cxr_chen import CvT2DistilGPT2MIMICXRChen

    checkpoint_path = Path(ckpt_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    config = load_yaml_config(config_path)
    config_kwargs = _config_kwargs_for_model(config, config_path)

    try:
        model = CvT2DistilGPT2MIMICXRChen.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            strict=strict_load,
            **config_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load CvT2DistilGPT2MIMICXRChen from checkpoint. "
            f"checkpoint={checkpoint_path}, config={config_path}, strict_load={strict_load}. "
            "Check that the YAML contains constructor parameters such as "
            "dataset_dir, ckpt_zoo_dir, mbatch_size, encoder_lr, decoder_lr, "
            "decoder_max_len, and num_test_beams."
        ) from exc

    model.eval()
    model.to(device)
    return model


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor values in a batch to the target device."""
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def safe_to_str_id(x: Any) -> str:
    """Convert a sample id from common collated forms into a stable string."""
    if torch.is_tensor(x):
        if x.numel() == 1:
            return str(x.detach().cpu().item())
        return "_".join(str(i) for i in x.detach().cpu().reshape(-1).tolist())
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return safe_to_str_id(x[0])
        return "_".join(safe_to_str_id(i) for i in x)
    return str(x)


def preview_text(text: Any, max_len: int = 120) -> str:
    """Return a compact one-line preview of text."""
    text = "" if text is None else str(text)
    text = " ".join(text.strip().split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def _select_split(model: Any, split: str):
    if split == "train":
        return model.train_set, model.train_dataloader(shuffle=False)
    if split == "val":
        return model.val_set, model.val_dataloader()
    if split == "test":
        return model.test_set, model.test_dataloader()
    raise ValueError(f"Unsupported split: {split}. Expected one of {sorted(_VALID_SPLITS)}.")


def _get_batch_field(batch: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in batch:
            return batch[name]
    raise KeyError(f"Batch is missing required field. Tried: {', '.join(names)}")


def _get_text_field(row: Dict[str, Any], field_names: Iterable[str], required: bool = True) -> Any:
    """Return the first present non-None field value from a JSON row."""
    for field_name in field_names:
        if field_name in row and row[field_name] is not None:
            return row[field_name]
    if required:
        raise KeyError(f"Row is missing required field. Tried: {', '.join(field_names)}")
    return None


def _iter_batch_values(values: Any) -> Iterable[Any]:
    if torch.is_tensor(values):
        return [values[i] for i in range(values.shape[0])]
    return list(values)


def _maybe_len(obj: Any) -> Any:
    try:
        return len(obj)
    except Exception:
        return "unknown"


def _extract_error_tags(score: Dict[str, Any]) -> List[str]:
    penalties = score.get("penalties", {})
    return [name for name, value in penalties.items() if float(value) > 0]


def _mean(values: Iterable[float]) -> Any:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _print_examples(prefix: str, ids, references, generated=None, limit: int = 3) -> None:
    count = min(limit, len(ids))
    for idx in range(count):
        print(f"[{prefix}] sample {idx}")
        print(f"  id: {safe_to_str_id(ids[idx])}")
        print(f"  reference: {preview_text(references[idx])}")
        if generated is not None:
            print(f"  generated: {preview_text(generated[idx])}")


def _print_pair_previews(previews: List[Dict[str, Any]], limit: int) -> None:
    count = min(limit, len(previews))
    for idx in range(count):
        row = previews[idx]
        print(f"[Preview] pair {idx}")
        print(f"  id: {row['id']}")
        print(f"  chosen: {preview_text(row['chosen'])}")
        print(f"  rejected: {preview_text(row['rejected'])}")
        print(f"  reward_rejected: {row['reward_rejected']}")
        print(f"  error_tags: {row['error_tags']}")


def generate_preds(args: argparse.Namespace) -> None:
    """Generate CE-model predictions for a requested split and write JSONL."""
    if args.split not in _VALID_SPLITS:
        raise ValueError(f"Unsupported split: {args.split}. Expected one of {sorted(_VALID_SPLITS)}.")

    device = torch.device(args.device)
    config = load_yaml_config(args.config)
    if args.batch_size is not None:
        config["mbatch_size"] = args.batch_size
    if int(config.get("num_workers", 0) or 0) <= 0:
        config["num_workers"] = 1

    model = load_model_from_config(
        config_path=args.config,
        ckpt_path=args.ckpt_path,
        device=device,
        strict_load=args.strict_load,
    )
    if args.batch_size is not None:
        model.mbatch_size = args.batch_size
    if int(getattr(model, "num_workers", 0) or 0) <= 0:
        model.num_workers = 1

    if args.split in {"train", "val"}:
        model.setup("fit")
    elif args.split == "test":
        model.setup("test")
    else:
        raise ValueError(f"Unsupported split: {args.split}")

    dataset, dataloader = _select_split(model, args.split)
    dataset_len = _maybe_len(dataset)
    batch_size = getattr(dataloader, "batch_size", getattr(model, "mbatch_size", "unknown"))
    out_path = Path(args.out_path).expanduser()

    print(f"[Info] split = {args.split}")
    print(f"[Info] num_examples = {dataset_len}")
    print(f"[Info] batch_size = {batch_size}")
    print(f"[Info] num_beams = {args.num_beams}")
    print(f"[Info] out_path = {out_path}")
    if args.split == "train" and dataset_len == _KNOWN_TEST_SIZE:
        print(
            "[WARNING] split=train dataset length is exactly 2210. This looks like the known "
            "test-set size in this project. Please verify the selected dataloader."
        )

    out_file = None
    if args.dry_run:
        print("[Dry-run] No output file will be written.")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("w", encoding="utf-8")

    first_ids = []
    first_refs = []
    first_generated = []
    num_written = 0
    truncated = False

    try:
        with torch.no_grad():
            for _batch_idx, batch in enumerate(dataloader):
                batch = move_batch_to_device(batch, device)
                if "encoder_images" not in batch:
                    raise KeyError('Batch is missing required field "encoder_images".')
                sample_ids = list(_iter_batch_values(_get_batch_field(batch, "id")))
                references = list(_iter_batch_values(_get_batch_field(batch, "labels", "label")))

                images = batch["encoder_images"]
                output_ids = model.generate(args.num_beams, images)
                generated = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                batch_count = min(len(sample_ids), len(references), len(generated))
                if batch_count != len(generated):
                    raise RuntimeError(
                        "Batch output size mismatch: "
                        f"ids={len(sample_ids)}, references={len(references)}, generated={len(generated)}"
                    )

                for i in range(batch_count):
                    sample_id = safe_to_str_id(sample_ids[i])
                    reference_text = str(references[i]).strip()
                    generated_text = str(generated[i]).strip()

                    if len(first_ids) < args.preview_n:
                        first_ids.append(sample_id)
                        first_refs.append(reference_text)
                        first_generated.append(generated_text)

                    if args.dry_run:
                        num_written += 1
                    else:
                        row = {
                            "id": sample_id,
                            "reference": reference_text,
                            "generated": generated_text,
                            "split": args.split,
                        }
                        out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        num_written += 1

                    if args.max_items is not None and num_written >= args.max_items:
                        truncated = True
                        break

                if args.dry_run:
                    print(f"[Dry-run] Generated one batch and previewed up to {args.preview_n} examples.")
                    break
                if truncated:
                    break
    finally:
        if out_file is not None:
            out_file.close()

    if truncated:
        print(f"[Info] Reached --max-items={args.max_items}; output is truncated.")

    if first_ids:
        _print_examples("Preview", first_ids, first_refs, first_generated, limit=args.preview_n)
    else:
        print("[Preview] No examples were generated.")

    if args.split == "train" and num_written == _KNOWN_TEST_SIZE:
        print(
            "[WARNING] split=train wrote exactly 2210 predictions. This looks like the known "
            "test-set size in this project. Please verify that generate-preds is really using "
            "train_dataloader(shuffle=False), not infer.py/test set."
        )

    print(f"[Done] split = {args.split}")
    print(f"[Done] num_examples = {dataset_len}")
    print(f"[Done] num_written = {num_written}")
    print(f"[Done] out_path = {out_path}")
    print(f"[Done] first_ids = {first_ids}")


def build_pairs_cmd(args: argparse.Namespace) -> None:
    """Build raw DPO pairs from generate-preds JSONL."""
    from tools.reward_evaluator import LiteClinicalRewardEvaluator

    pred_path = Path(args.pred_path).expanduser()
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction JSONL file not found: {pred_path}")
    out_path = Path(args.out_path).expanduser()

    evaluator = LiteClinicalRewardEvaluator(
        min_len=args.reward_min_len,
        max_len=args.reward_max_len,
        repeat_ngram=args.reward_repeat_ngram,
        short_penalty=args.reward_short_penalty,
        repeat_penalty=args.reward_repeat_penalty,
        normal_template_penalty=args.reward_normal_template_penalty,
    )

    rows = load_jsonl(str(pred_path))
    total_rows = len(rows)
    if total_rows == _FULL_MIMIC_TRAIN_SIZE:
        print("[Info] This looks like full MIMIC train split generated by generate-preds.")
    if total_rows == _KNOWN_TEST_SIZE:
        print("[WARNING] Input has 2210 rows, which looks like test set size. Do not use test pairs for DPO training.")

    id_fields = [args.id_field] if args.id_field else _ID_FIELDS
    reference_fields = [args.reference_field] if args.reference_field else _REFERENCE_FIELDS
    generated_fields = [args.generated_field] if args.generated_field else _GENERATED_FIELDS

    skipped_empty_id = 0
    skipped_empty_reference = 0
    skipped_empty_generated = 0
    skipped_identical = 0
    skipped_reward_gap = 0
    split_counts = Counter()
    error_tag_counts = Counter()
    reward_rejected_values = []
    previews = []
    written_pairs = 0
    truncated = False

    out_file = None
    if args.dry_run:
        print("[Dry-run] No output file will be written.")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("w", encoding="utf-8")

    try:
        for row_idx, row in enumerate(rows, start=1):
            try:
                raw_id = _get_text_field(row, id_fields, required=True)
                raw_chosen = _get_text_field(row, reference_fields, required=True)
                raw_rejected = _get_text_field(row, generated_fields, required=True)
            except KeyError as exc:
                raise KeyError(f"{exc} at prediction row {row_idx} in {pred_path}") from exc

            sample_id = safe_to_str_id(raw_id).strip()
            chosen = str(raw_chosen).strip()
            rejected = str(raw_rejected).strip()
            split = row.get("split")

            if not sample_id:
                skipped_empty_id += 1
                continue
            if not chosen:
                skipped_empty_reference += 1
                continue
            if not rejected:
                skipped_empty_generated += 1
                continue
            if args.skip_identical and chosen == rejected:
                skipped_identical += 1
                continue

            score = evaluator.score_one(rejected, chosen)
            reward_rejected = float(score["reward"])
            reward_chosen = 1.0
            reward_gap = reward_chosen - reward_rejected
            if args.min_reward_gap > 0 and reward_gap < args.min_reward_gap:
                skipped_reward_gap += 1
                continue

            error_tags = _extract_error_tags(score)
            out_row = {
                "id": sample_id,
                "chosen": chosen,
                "rejected": rejected,
                "reward_chosen": reward_chosen,
                "reward_rejected": reward_rejected,
                "error_tags": error_tags,
                "split": split,
                "source": {
                    "chosen": "prediction_reference",
                    "rejected": "prediction",
                },
            }
            if args.include_reward_details:
                out_row["reward_details"] = score

            if len(previews) < args.preview_n:
                previews.append(out_row)

            split_counts[str(split)] += 1
            error_tag_counts.update(error_tags)
            reward_rejected_values.append(reward_rejected)

            if not args.dry_run:
                out_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written_pairs += 1

            if args.max_items is not None and written_pairs >= args.max_items:
                truncated = True
                break
    finally:
        if out_file is not None:
            out_file.close()

    print(f"[Info] pred_path = {pred_path}")
    print(f"[Info] out_path = {out_path}")
    print(f"[Info] total_prediction_rows = {total_rows}")
    print(f"[Info] written_pairs = {written_pairs}")
    print(f"[Info] skipped_empty_id = {skipped_empty_id}")
    print(f"[Info] skipped_empty_reference = {skipped_empty_reference}")
    print(f"[Info] skipped_empty_generated = {skipped_empty_generated}")
    print(f"[Info] skipped_identical = {skipped_identical}")
    print(f"[Info] skipped_reward_gap = {skipped_reward_gap}")
    print(f"[Info] split_counts = {dict(split_counts)}")
    print(f"[Info] reward_rejected_mean = {_mean(reward_rejected_values)}")
    print(f"[Info] reward_rejected_min = {min(reward_rejected_values) if reward_rejected_values else None}")
    print(f"[Info] reward_rejected_max = {max(reward_rejected_values) if reward_rejected_values else None}")
    print(f"[Info] error_tag_counts = {dict(error_tag_counts)}")
    if truncated:
        print(f"[Info] Reached --max-items={args.max_items}; output is truncated.")
    if previews:
        _print_pair_previews(previews, args.preview_n)
    else:
        print("[Preview] No pairs were built.")
    if args.dry_run:
        print("[Dry-run] Finished without writing output.")
    print(f"[Done] written_pairs = {written_pairs}")
    print(f"[Done] out_path = {out_path}")


def _not_implemented(command: str) -> None:
    raise NotImplementedError(f"{command} will be implemented in the next step.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utilities for building DPO JSONL data for CvT2DistilGPT2 RRG."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate-preds",
        help="Generate split-aware CE model predictions as id/reference/generated/split JSONL.",
    )
    generate.add_argument("--config", required=True, help="Path to YAML config.")
    generate.add_argument("--ckpt-path", required=True, help="Path to model checkpoint.")
    generate.add_argument("--split", required=True, choices=sorted(_VALID_SPLITS), help="Dataset split.")
    generate.add_argument("--out-path", required=True, help="Output JSONL path.")
    generate.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    generate.add_argument("--batch-size", type=int, default=None, help="Override model mini-batch size.")
    generate.add_argument("--num-beams", type=int, default=1, help="Number of beams for generation.")
    generate.add_argument("--max-items", type=int, default=None, help="Optional maximum examples to emit.")
    generate.add_argument("--strict-load", type=_str_to_bool, default=False, help="Use strict checkpoint loading.")
    generate.add_argument("--dry-run", action="store_true", help="Generate previews without writing a file.")
    generate.add_argument("--preview-n", type=int, default=3, help="Number of examples to preview.")
    generate.set_defaults(func=generate_preds)

    build_pairs_parser = subparsers.add_parser(
        "build-pairs",
        help="Build raw DPO pairs from generate-preds JSONL.",
    )
    build_pairs_parser.add_argument("--pred-path", required=True, help="Input predictions JSONL path.")
    build_pairs_parser.add_argument("--out-path", required=True, help="Output DPO pair JSONL path.")
    build_pairs_parser.add_argument("--id-field", default=None, help="Override input ID field name.")
    build_pairs_parser.add_argument("--reference-field", default=None, help="Override input reference/chosen field name.")
    build_pairs_parser.add_argument("--generated-field", default=None, help="Override input generated/rejected field name.")
    build_pairs_parser.add_argument("--skip-identical", type=_str_to_bool, default=True, help="Skip identical chosen/rejected texts.")
    build_pairs_parser.add_argument("--min-reward-gap", type=float, default=0.0, help="Minimum reward_chosen - reward_rejected gap.")
    build_pairs_parser.add_argument("--max-items", type=int, default=None, help="Optional maximum pairs to emit.")
    build_pairs_parser.add_argument("--dry-run", action="store_true", help="Preview and report stats without writing output.")
    build_pairs_parser.add_argument("--preview-n", type=int, default=3, help="Number of built pairs to preview.")
    build_pairs_parser.add_argument(
        "--include-reward-details",
        type=_str_to_bool,
        default=False,
        help="Include full reward_details in each output row.",
    )
    build_pairs_parser.add_argument("--reward-min-len", type=int, default=5, help="Reward evaluator minimum token length.")
    build_pairs_parser.add_argument("--reward-max-len", type=int, default=120, help="Reward evaluator maximum token length.")
    build_pairs_parser.add_argument("--reward-repeat-ngram", type=int, default=3, help="Repeated n-gram size for reward penalty.")
    build_pairs_parser.add_argument("--reward-short-penalty", type=float, default=0.15, help="Short/long text reward penalty.")
    build_pairs_parser.add_argument("--reward-repeat-penalty", type=float, default=0.2, help="Repeated text reward penalty scale.")
    build_pairs_parser.add_argument(
        "--reward-normal-template-penalty",
        type=float,
        default=0.1,
        help="Normal-template reward penalty.",
    )
    build_pairs_parser.set_defaults(func=build_pairs_cmd)

    add_ref_logps = subparsers.add_parser("add-ref-logps", help="Reserved for future reference log-probs.")
    add_ref_logps.set_defaults(func=lambda args: _not_implemented("add-ref-logps"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
