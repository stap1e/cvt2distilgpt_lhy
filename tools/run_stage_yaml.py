"""Run a stages.py entrypoint directly from a YAML config.

This wrapper bypasses dlhpcstarter's Hydra config loading path while preserving the
repository's existing stages.stages(args) training/test flow.
"""

import argparse
import os
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_FALSEY_STRINGS = {"false", "0", "no", "n", "none", "null", ""}
_TRUEY_STRINGS = {"true", "1", "yes", "y"}
_NUMERIC_STRING_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in _TRUEY_STRINGS:
        return True
    if lowered in _FALSEY_STRINGS:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def _load_yaml(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return dict(config)


def _coerce_numeric_strings(value: Any) -> Any:
    """Match Hydra/OmegaConf-style numeric coercion for YAML 1.1 scientific notation strings."""
    if isinstance(value, dict):
        return {key: _coerce_numeric_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_numeric_strings(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if _NUMERIC_STRING_RE.match(stripped):
            try:
                if any(char in stripped for char in ".eE"):
                    return float(stripped)
                return int(stripped)
            except ValueError:
                return value
    return value


def _default_args() -> Dict[str, Any]:
    return {
        "accumulated_mbatch_size": None,
        "auto_resubmit": False,
        "auto_resubmit_method": "timeout",
        "callbacks": None,
        "compile": None,
        "config_dir": None,
        "config_file_name": None,
        "config_full_path": None,
        "config_name": None,
        "cuda_visible_devices": None,
        "debug": False,
        "deterministic": True,
        "devices": 1,
        "divergence_threshold": None,
        "early_stopping": False,
        "email": None,
        "enable_progress_bar": None,
        "every_n_epochs": 1,
        "every_n_train_steps": None,
        "exp_dir": None,
        "exp_dir_trial": None,
        "fast_dev_run": False,
        "float32_matmul_precision": None,
        "loggers": None,
        "manager_script_path": None,
        "mbatch_size": None,
        "memory": None,
        "min_delta": 0.0,
        "mlflow_logger": False,
        "monitor": None,
        "monitor_mode": "min",
        "neptune_api_key": None,
        "neptune_mode": "async",
        "neptune_username": None,
        "no_cpus_per_task": False,
        "no_gpus_per_node": False,
        "no_ntasks_per_node": False,
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
        "num_workers": 1,
        "one_epoch_only": False,
        "other_exp_dir": None,
        "other_monitor": None,
        "other_monitor_mode": None,
        "patience": 0,
        "plugins": None,
        "precision": None,
        "qos": None,
        "resume_ckpt_path": None,
        "resume_epoch": None,
        "resume_last": False,
        "save_top_k": 1,
        "search_space": None,
        "search_space_ignore_keys": [],
        "seed": None,
        "srun_options": None,
        "stages_definition": "stages",
        "stages_module": "stages",
        "strategy": "auto",
        "submit": False,
        "task": None,
        "test": False,
        "test_ckpt_path": None,
        "test_epoch": None,
        "test_without_ckpt": False,
        "time_limit": None,
        "train": False,
        "trial": 0,
        "validate": False,
        "venv_path": None,
        "warm_start_ckpt_path": None,
        "warm_start_ckpt_path_strict": False,
        "warm_start_modules": False,
        "weights_summary": None,
        "work_dir": None,
    }


def _merge_args(cli_args: argparse.Namespace) -> Namespace:
    config_path = Path(cli_args.config).expanduser().resolve()
    config = _coerce_numeric_strings(_load_yaml(str(config_path)))

    merged = _default_args()
    merged.update(config)

    merged["task"] = cli_args.task
    merged["config"] = str(config_path)
    merged["config_file_name"] = str(config_path)
    merged["config_full_path"] = str(config_path)
    merged["config_name"] = config_path.stem
    merged["config_dir"] = str(config_path.parent)
    merged["work_dir"] = str(Path(cli_args.work_dir).expanduser().resolve()) if cli_args.work_dir else str(REPO_ROOT)
    merged["train"] = bool(cli_args.train)
    merged["test"] = bool(cli_args.test)
    merged["stages_module"] = cli_args.stages_module
    merged["stages_definition"] = cli_args.stages_definition

    if cli_args.trial is not None:
        merged["trial"] = cli_args.trial
    elif merged.get("trial") is None:
        merged["trial"] = 0

    if cli_args.exp_dir_trial is not None:
        merged["exp_dir_trial"] = cli_args.exp_dir_trial
    elif merged.get("exp_dir_trial") is None:
        exp_dir = merged.get("exp_dir")
        if exp_dir is None:
            raise ValueError("exp_dir must be set in the YAML config or via --exp-dir-trial.")
        merged["exp_dir_trial"] = os.path.join(
            str(exp_dir),
            str(merged["task"]),
            str(merged["config_name"]),
            "trial_" + str(merged["trial"]),
        )

    override_keys = (
        "devices",
        "num_nodes",
        "num_workers",
        "max_epochs",
        "fast_dev_run",
        "resume_last",
        "resume_epoch",
        "resume_ckpt_path",
        "warm_start_ckpt_path",
        "cuda_visible_devices",
    )
    for key in override_keys:
        value = getattr(cli_args, key, None)
        if value is not None:
            merged[key] = value

    if not merged["train"] and not merged["test"]:
        raise ValueError("Specify at least one of --train or --test.")

    if int(merged.get("num_workers", 0) or 0) == 0:
        merged["prefetch_factor"] = None
    else:
        merged.setdefault("prefetch_factor", 2)

    Path(merged["exp_dir_trial"]).mkdir(parents=True, exist_ok=True)
    return Namespace(**merged)


def _patch_stages_config_loader(stages_module: Any) -> None:
    """Prevent stages.stages() from calling dlhpcstarter's Hydra loader again."""
    def _identity_loader(args: Namespace, print_args: bool = False):
        if print_args:
            print(f"args: {args.__dict__}")
        return args, args

    stages_module.load_config_and_update_args = _identity_loader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a stages.py entrypoint directly from a YAML config.")
    parser.add_argument("--task", "-t", required=True, help="Task/experiment name.")
    parser.add_argument("--config", "-c", required=True, help="YAML config path.")
    parser.add_argument("--stages-module", "--stages_module", default="stages", help="Module containing stages().")
    parser.add_argument("--stages-definition", "--stages_definition", default="stages", help="Stages function name.")
    parser.add_argument("--train", action="store_true", help="Run training.")
    parser.add_argument("--test", action="store_true", help="Run testing.")
    parser.add_argument("--trial", type=int, default=None, help="Override trial number.")
    parser.add_argument("--work-dir", "--work_dir", default=None, help="Working directory for imports and outputs.")
    parser.add_argument("--exp-dir-trial", "--exp_dir_trial", default=None, help="Override experiment trial directory.")
    parser.add_argument("--devices", type=int, default=None, help="Override devices.")
    parser.add_argument("--num-nodes", "--num_nodes", type=int, default=None, help="Override num_nodes.")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=None, help="Override num_workers.")
    parser.add_argument("--max-epochs", "--max_epochs", type=int, default=None, help="Override max_epochs.")
    parser.add_argument("--fast-dev-run", "--fast_dev_run", action="store_true", default=None, help="Enable Lightning fast_dev_run.")
    parser.add_argument("--resume-last", "--resume_last", type=_str_to_bool, default=None, help="Override resume_last.")
    parser.add_argument("--resume-epoch", "--resume_epoch", type=int, default=None, help="Override resume_epoch.")
    parser.add_argument("--resume-ckpt-path", "--resume_ckpt_path", default=None, help="Override resume_ckpt_path.")
    parser.add_argument("--warm-start-ckpt-path", "--warm_start_ckpt_path", default=None, help="Override warm_start_ckpt_path.")
    parser.add_argument("--cuda-visible-devices", "--cuda_visible_devices", default=None, help="Set CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--print-args-only", action="store_true", help="Build and print args without importing/running stages.")
    return parser


def main() -> None:
    parser = build_parser()
    cli_args = parser.parse_args()
    args = _merge_args(cli_args)

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        print(f"CUDA_VISIBLE_DEVICES: {args.cuda_visible_devices}")

    if cli_args.print_args_only:
        for key in sorted(vars(args)):
            print(f"{key}: {getattr(args, key)}")
        return

    from dlhpcstarter.utils import importer

    stages_module = __import__(args.stages_module)
    _patch_stages_config_loader(stages_module)
    stages_fnc = importer(definition=args.stages_definition, module=args.stages_module)
    stages_fnc(args)


if __name__ == "__main__":
    main()
