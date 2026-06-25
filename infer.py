#!/usr/bin/env python3
"""Standalone inference for the CvT-to-DistilGPT2 baselines.

This script intentionally avoids dlhpcstarter and Lightning Trainer.test(). It
loads an existing test config/checkpoint, builds the model's test dataloader,
calls model.generate() directly, and streams predictions to JSONL.
"""

import argparse
import importlib
import json
import os
import random
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a CvT2DistilGPT2 checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a test YAML config.")
    parser.add_argument("--ckpt-path", default=None, help="Override config test_ckpt_path.")
    parser.add_argument("--output-path", default=None, help="JSONL output path.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config mbatch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override config num_workers.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Override config prefetch_factor.")
    parser.add_argument("--num-beams", type=int, default=None, help="Override config num_test_beams.")
    parser.add_argument("--decoder-max-len", type=int, default=None, help="Override config decoder_max_len.")
    parser.add_argument("--max-samples", type=int, default=None, help="Stop after this many samples.")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config


def apply_overrides(config: Dict[str, Any], cli_args: argparse.Namespace) -> Namespace:
    config = dict(config)

    if cli_args.ckpt_path is not None:
        config["test_ckpt_path"] = cli_args.ckpt_path
    if cli_args.batch_size is not None:
        config["mbatch_size"] = cli_args.batch_size
    if cli_args.num_workers is not None:
        config["num_workers"] = cli_args.num_workers
    if cli_args.prefetch_factor is not None:
        config["prefetch_factor"] = cli_args.prefetch_factor
    if cli_args.num_beams is not None:
        config["num_test_beams"] = cli_args.num_beams
    if cli_args.decoder_max_len is not None:
        config["decoder_max_len"] = cli_args.decoder_max_len

    config.setdefault("warm_start_modules", False)
    config.setdefault("prefetch_factor", 5)
    config.setdefault("num_workers", 1)
    config.setdefault("exp_dir_trial", config.get("exp_dir", "."))

    # Existing model dataloaders unconditionally pass prefetch_factor, which is
    # invalid when num_workers == 0 in PyTorch.
    if int(config.get("num_workers", 0)) == 0:
        print("[Warn] num_workers=0 is incompatible with the current model test_dataloader; using num_workers=1.")
        config["num_workers"] = 1

    required = [
        "module",
        "definition",
        "test_ckpt_path",
        "dataset_dir",
        "ckpt_zoo_dir",
        "exp_dir_trial",
        "mbatch_size",
        "decoder_max_len",
        "num_test_beams",
    ]
    missing = [key for key in required if key not in config or config[key] is None]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")

    return Namespace(**config)


def import_model(module_name: str, definition: str):
    module = importlib.import_module(module_name)
    return getattr(module, definition)


def resolve_device(device_name: Optional[str]) -> torch.device:
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")
    return device


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_output_path(args: Namespace, config_path: str) -> Path:
    config_stem = Path(config_path).stem
    ckpt_stem = Path(str(args.test_ckpt_path)).stem.replace("=", "_")
    output_dir = Path(args.exp_dir_trial) / "inference"
    return output_dir / f"{config_stem}_{ckpt_stem}.jsonl"


def move_tensor_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def batch_values(value: Any, batch_size: int) -> List[Any]:
    if value is None:
        return [None] * batch_size
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) == batch_size:
            return value
        return [value] * batch_size
    return [value] * batch_size


def write_records(
    output_path: Path,
    ids: Iterable[Any],
    predictions: Iterable[str],
    references: Iterable[Any],
    image_filepaths: Iterable[Any],
    ckpt_path: str,
    num_beams: int,
    limit: Optional[int],
    written: int,
) -> int:
    with output_path.open("a", encoding="utf-8") as f:
        for sample_id, prediction, reference, image_path in zip(ids, predictions, references, image_filepaths):
            if limit is not None and written >= limit:
                break
            record = {
                "id": sample_id,
                "prediction": prediction,
                "reference": reference,
                "ckpt_path": ckpt_path,
                "num_beams": num_beams,
            }
            if image_path is not None:
                record["image_filepaths"] = image_path
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def run_inference(cli_args: argparse.Namespace) -> Path:
    config = load_config(cli_args.config)
    args = apply_overrides(config, cli_args)
    device = resolve_device(cli_args.device)
    seed_everything(getattr(args, "seed", None))

    output_path = Path(cli_args.output_path) if cli_args.output_path else default_output_path(args, cli_args.config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    TaskModel = import_model(args.module, args.definition)

    print(f"[Info] Loading checkpoint: {args.test_ckpt_path}")
    model = TaskModel.load_from_checkpoint(
        checkpoint_path=args.test_ckpt_path,
        strict=False,
        **vars(args),
    )
    model.to(device)
    model.eval()

    print("[Info] Building test dataloader.")
    model.setup("test")
    loader = model.test_dataloader()

    written = 0
    print(f"[Info] Writing predictions to: {output_path}")
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            images = move_tensor_to_device(batch["encoder_images"], device)
            output_ids = model.generate(args.num_test_beams, images)
            predictions = model.tokenizer.batch_decode(output_ids.detach().cpu(), skip_special_tokens=True)

            batch_size = len(predictions)
            ids = batch_values(batch.get("id"), batch_size)
            references = batch_values(batch.get("labels"), batch_size)
            image_filepaths = batch_values(batch.get("image_filepaths"), batch_size)

            written = write_records(
                output_path=output_path,
                ids=ids,
                predictions=predictions,
                references=references,
                image_filepaths=image_filepaths,
                ckpt_path=str(args.test_ckpt_path),
                num_beams=int(args.num_test_beams),
                limit=cli_args.max_samples,
                written=written,
            )

            print(f"[Info] Processed batch {batch_idx + 1}; total predictions: {written}")
            if cli_args.max_samples is not None and written >= cli_args.max_samples:
                break

    print(f"[Info] Done. Wrote {written} predictions.")
    return output_path


def main() -> None:
    cli_args = parse_args()
    run_inference(cli_args)


if __name__ == "__main__":
    main()
