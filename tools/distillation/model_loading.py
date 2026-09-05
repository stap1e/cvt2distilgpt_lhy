"""Shared checkpoint loading for the offline distillation CLIs.

Generalises ``tools/dpo_json.load_model_from_config`` so that COVAR-V2
checkpoints (which contain the evidence planner) are loaded through the
correct subclass instead of the baseline class. Torch imports happen lazily
inside functions so module import stays CPU-test friendly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_yaml_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config


_MODEL_REGISTRY = {
    "cvt2distilgpt2_mimic_cxr_chen": ("cvt2distilgpt2_mimic_cxr_chen", "CvT2DistilGPT2MIMICXRChen"),
    "cvt2distilgpt2_mimic_cxr_visual_grounded": (
        "cvt2distilgpt2_mimic_cxr_visual_grounded",
        "CvT2DistilGPT2MIMICXRVisualGrounded",
    ),
    "cvt2distilgpt2_mimic_cxr_visual_grounded_v2": (
        "cvt2distilgpt2_mimic_cxr_visual_grounded_v2",
        "CvT2DistilGPT2MIMICXRVisualGroundedV2",
    ),
}


def config_kwargs_for_offline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare constructor kwargs for offline (non-training) model use."""
    import copy

    kwargs = copy.deepcopy(config)
    kwargs["warm_start_modules"] = False
    if "exp_dir_trial" not in kwargs:
        kwargs["exp_dir_trial"] = str(Path(kwargs.get("exp_dir", "./outputs")).expanduser())
    if int(kwargs.get("num_workers", 0) or 0) <= 0:
        kwargs["num_workers"] = 1
    # Offline CLIs never train; force the CE path so the module constructor
    # validation accepts the config regardless of the staged train_mode.
    kwargs["train_mode"] = "ce"
    kwargs["dpo_pair_path"] = None
    kwargs.pop("module", None)
    kwargs.pop("definition", None)
    kwargs.pop("warm_start_ckpt_path", None)
    return kwargs


def load_model_from_config(
    config_path: str,
    ckpt_path: str,
    device,
    strict_load: bool = False,
):
    """Load the model class selected by the config's ``module`` key."""
    import torch

    checkpoint = Path(ckpt_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

    config = load_yaml_config(config_path)
    module_name = str(config.get("module", "cvt2distilgpt2_mimic_cxr_chen"))
    if module_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Config module {module_name!r} is not in the offline registry "
            f"{sorted(_MODEL_REGISTRY)}."
        )
    import_module, class_name = _MODEL_REGISTRY[module_name]
    module = __import__(import_module)
    model_class = getattr(module, class_name)

    kwargs = config_kwargs_for_offline(config)
    model = model_class.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        strict=strict_load,
        **kwargs,
    )
    model.eval()
    model.to(device)
    return model


def load_student_tokenizer(config: Dict[str, Any]):
    """Load the student GPT2 fast tokenizer from the local ckpt zoo."""
    import transformers

    ckpt_zoo = config.get("ckpt_zoo_dir")
    if not ckpt_zoo:
        raise ValueError("Config must define ckpt_zoo_dir to load the student tokenizer.")
    decoder_path = Path(ckpt_zoo).expanduser() / "distilbert" / "distilgpt2"
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained(
        str(decoder_path), local_files_only=True
    )
    tokenizer.add_special_tokens({"bos_token": "[BOS]", "pad_token": "[PAD]"})
    return tokenizer
