"""Per-run algorithmic record writer (``new_lab`` experiment records).

The recorder appends one JSON line per epoch-level event and re-renders a
compact markdown table so that after every training run the record can be
pasted into ``docs/new_lab/EXPERIMENTS.md`` and analysed.

Recording policy (per repository convention for this branch):
- IN: loss components (CE, evidence, concept, EOS, GKD/DPO terms), validation
  metrics (CIDEr, BLEU, CheXbert F1, temporal-claim rate, ...), pool/round
  events, key hyperparameters.
- OUT: anything about compute consumption — wall-clock time, throughput, GPU
  memory, hardware. The whitelist below enforces this.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

# Substrings that must never appear in recorded metric keys.
_EXCLUDED_SUBSTRINGS = ("time", "sec", "gpu", "mem", "device", "throughput", "fps")
# Metric keys are recorded only if their prefix matches.
_PREFIX_WHITELIST = ("train_", "val_", "test_")


def is_recordable_metric(key: str) -> bool:
    lowered = key.lower()
    if any(bad in lowered for bad in _EXCLUDED_SUBSTRINGS):
        return False
    return lowered.startswith(_PREFIX_WHITELIST)


def _as_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


class RunRecorder:
    """Append-only JSONL + markdown run record inside the trial directory."""

    def __init__(
        self,
        exp_dir: str,
        run_name: str,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.exp_dir = Path(exp_dir).expanduser()
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.jsonl_path = self.exp_dir / "new_lab_run_record.jsonl"
        self.markdown_path = self.exp_dir / "new_lab_run_record.md"
        self._events: list = []

        self._write_event(
            {
                "event": "run_start",
                "run_name": run_name,
                "meta": dict(meta or {}),
            }
        )
        self._render_markdown()

    # ------------------------------------------------------------------

    def record_epoch_metrics(
        self,
        epoch: int,
        kind: str,
        metrics: Mapping[str, Any],
    ) -> Dict[str, float]:
        """Filter ``metrics`` to the whitelist and append one record."""
        recorded = {}
        for key, value in metrics.items():
            if not is_recordable_metric(key):
                continue
            number = _as_float(value)
            if number is None:
                continue
            recorded[str(key)] = number
        if not recorded:
            return recorded

        self._write_event(
            {
                "event": "epoch_metrics",
                "epoch": int(epoch),
                "kind": kind,
                "metrics": recorded,
            }
        )
        self._render_markdown()
        return recorded

    def record_event(self, title: str, fields: Optional[Mapping[str, Any]] = None):
        """Record a discrete algorithmic event (e.g. GKD pool loaded)."""
        self._write_event(
            {"event": "custom", "title": title, "fields": dict(fields or {})}
        )
        self._render_markdown()

    # ------------------------------------------------------------------

    def _write_event(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        # Local timestamp only orders events; it is analysis metadata, not a
        # compute-consumption metric.
        event["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events.append(event)

    def _render_markdown(self) -> None:
        lines = [f"# new_lab run record — {self.run_name}", ""]

        meta_events = [e for e in self._events if e.get("event") == "run_start"]
        if meta_events and meta_events[-1].get("meta"):
            lines.append("## Run metadata")
            lines.append("")
            for key, value in meta_events[-1]["meta"].items():
                lines.append(f"- {key}: `{value}`")
            lines.append("")

        custom = [e for e in self._events if e.get("event") == "custom"]
        if custom:
            lines.append("## Events")
            lines.append("")
            for event in custom:
                fields = event.get("fields") or {}
                suffix = (
                    " — " + ", ".join(f"{k}={v}" for k, v in fields.items())
                    if fields
                    else ""
                )
                lines.append(f"- [{event.get('ts')}] {event.get('title')}{suffix}")
            lines.append("")

        epoch_events = [e for e in self._events if e.get("event") == "epoch_metrics"]
        if epoch_events:
            keys: list = []
            for event in epoch_events:
                for key in event["metrics"]:
                    if key not in keys:
                        keys.append(key)
            lines.append("## Epoch metrics")
            lines.append("")
            header = "| epoch | kind | " + " | ".join(keys) + " |"
            separator = "|---|---|" + "|".join(["---"] * len(keys)) + "|"
            lines.append(header)
            lines.append(separator)
            for event in epoch_events:
                cells = [
                    _format_number(event["metrics"].get(key)) for key in keys
                ]
                lines.append(
                    f"| {event.get('epoch')} | {event.get('kind')} | "
                    + " | ".join(cells)
                    + " |"
                )
            lines.append("")

        self.markdown_path.write_text("\n".join(lines), encoding="utf-8")


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.6g}"


def collect_callback_metrics(
    callback_metrics: Mapping[str, Any],
) -> Dict[str, float]:
    """Whitelisted float view over ``trainer.callback_metrics``."""
    collected = {}
    for key, value in callback_metrics.items():
        if not is_recordable_metric(str(key)):
            continue
        number = _as_float(value)
        if number is not None:
            collected[str(key)] = number
    return collected
