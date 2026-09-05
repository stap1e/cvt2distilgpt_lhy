"""Teacher interfaces for offline distillation artefacts.

Teachers run ONLY inside the offline CLIs (``teacher_rewrite.py``,
``teacher_score.py``), never inside training, so the training GPU is never
shared with teacher inference. Three implementations:

- ``MockTeacher`` — deterministic, dependency-light, reference-driven.
  Intended for pipeline smoke tests and protocol debugging ONLY. It is NOT a
  research teacher; every report generated with it must be labelled as such
  in the experiment record.
- ``FileTeacher`` — reads teacher outputs produced elsewhere (a separate
  vLLM/transformers environment or another machine). This is the recommended
  bridge for real teachers such as MedGemma: run the teacher there, ship the
  JSONL here.
- ``LocalVLMTeacher`` — reference implementation of a real local multimodal
  teacher (image-grounded rewriting, scoring, ranking). It requires a modern
  ``transformers`` (>= 4.5x), which this repository's pinned environment does
  NOT provide — run it from a separate virtual environment.

All methods are torch-free at module import; heavy imports happen lazily.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tools.distillation.rollout_store import read_jsonl

_MOCK_CLINICAL_TOKENS = (
    "effusion", "edema", "atelectasis", "consolidation", "pneumonia",
    "pneumothorax", "cardiomegaly", "nodule", "mass", "fracture", "opacity",
    "opacities", "infiltrate", "tube", "catheter", "line", "device",
    "pacer", "pacemaker", "hernia", "emphysema", "fibrosis", "lesion",
)


class TeacherBase(ABC):
    """Common teacher protocol shared by rewrite / score / rank artefacts."""

    name: str = "base"

    @abstractmethod
    def rewrite_report(self, reference: str, image_path: Optional[str] = None) -> str:
        """Form A: produce a current-observable style rewrite of the report."""

    @abstractmethod
    def score_tokens(
        self,
        text: str,
        reference: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Dict:
        """Form B: teacher token log-probs for an arbitrary student text.

        Returns ``{"teacher_token_logps": [...], "teacher_token_offsets":
        [[start, end], ...]}`` where both lists describe the TEACHER's own
        tokenisation of ``text``; the caller aligns them onto student tokens
        via ``tools.distillation.token_align``.
        """

    @abstractmethod
    def rank_score(
        self,
        text: str,
        reference: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> float:
        """Form C: a single scalar quality score used for pairwise ranking."""


# ----------------------------------------------------------------------


class MockTeacher(TeacherBase):
    """Deterministic smoke-test teacher (NOT for formal experiments).

    - ``rewrite_report`` is an honest identity: the pipeline effect of Form A
      can still be audited because the current-observable filter runs after it.
    - ``score_tokens`` rewards tokens that also occur in the reference and
      penalises clinical-sounding tokens that do not — a crude stand-in for a
      grounded teacher that catches hallucinated findings.
    - ``rank_score`` combines reference token-F1 with the lexical reward of
      ``tools.reward_evaluator.LiteClinicalRewardEvaluator``.
    """

    name = "mock"

    def __init__(self, in_ref_logp: float = -0.20, off_ref_logp: float = -0.80,
                 off_ref_clinical_logp: float = -2.50) -> None:
        self.in_ref_logp = float(in_ref_logp)
        self.off_ref_logp = float(off_ref_logp)
        self.off_ref_clinical_logp = float(off_ref_clinical_logp)
        from tools.reward_evaluator import LiteClinicalRewardEvaluator

        self.reward_evaluator = LiteClinicalRewardEvaluator()

    def rewrite_report(self, reference: str, image_path: Optional[str] = None) -> str:
        return str(reference)

    def _tokenize_words(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        tokens = []
        for match in re.finditer(r"\S+", str(text)):
            word = match.group(0)
            tokens.append((word, (match.start(), match.end())))
        return tokens

    def score_tokens(
        self,
        text: str,
        reference: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Dict:
        reference_tokens = set(
            re.findall(r"[a-z0-9]+", str(reference or "").lower())
        )
        logps: List[float] = []
        offsets: List[List[int]] = []
        for word, (start, end) in self._tokenize_words(text):
            pieces = re.findall(r"[a-z0-9]+", word.lower())
            if not pieces:
                continue
            for piece in pieces:
                if piece in reference_tokens:
                    logp = self.in_ref_logp
                elif piece in _MOCK_CLINICAL_TOKENS:
                    logp = self.off_ref_clinical_logp
                else:
                    logp = self.off_ref_logp
                logps.append(logp)
            offsets.append([start, end])
        return {
            "teacher_token_logps": logps,
            "teacher_token_offsets": offsets,
        }

    def rank_score(
        self,
        text: str,
        reference: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> float:
        score = self.reward_evaluator.score_one(text, reference)
        return float(score["reward"])


# ----------------------------------------------------------------------


class FileTeacher(TeacherBase):
    """Serve teacher outputs that were produced in a separate environment.

    Expected JSONL schemas (one line per example):

    - rewrite: ``{"id": ..., "rewrite": str}``
    - scores:  ``{"id": ..., "teacher_token_logps": [float], 
    "teacher_token_offsets": [[s, e]]}`` (teacher tokenisation of the text
    being scored; the text itself is identified by id+k in rank files or by
    exact text match for score files)
    - rank:    ``{"id": ..., "scores": {"<text-hash-or-key>": float}}`` — for
    ranking artefacts we match rows by (id, text) so a key of the exact text
    string is the simplest contract.

    All modes fall back to MockTeacher behaviour (with a warning counter)
    when an id is missing, so a partially-generated teacher file never
    silently truncates a training pool.
    """

    name = "file"

    def __init__(self, teacher_file: str, mode: str = "auto") -> None:
        self.path = Path(teacher_file).expanduser()
        self.mode = mode
        self.rows = {str(row["id"]): row for row in read_jsonl(str(self.path))}
        self.fallback = MockTeacher()
        self.fallback_hits = 0

    def _row_for(self, key_id: str) -> Optional[dict]:
        row = self.rows.get(str(key_id))
        if row is None:
            self.fallback_hits += 1
        return row

    def rewrite_report(self, reference: str, image_path: Optional[str] = None,
                       key_id: Optional[str] = None) -> str:
        if key_id is not None:
            row = self._row_for(key_id)
            if row is not None and row.get("rewrite"):
                return str(row["rewrite"])
        return self.fallback.rewrite_report(reference, image_path)

    def score_tokens(
        self,
        text: str,
        reference: Optional[str] = None,
        image_path: Optional[str] = None,
        key_id: Optional[str] = None,
    ) -> Dict:
        if key_id is not None:
            row = self._row_for(key_id)
            if row is not None and "teacher_token_logps" in row:
                return {
                    "teacher_token_logps": list(row["teacher_token_logps"]),
                    "teacher_token_offsets": [
                        list(pair) for pair in row.get("teacher_token_offsets", [])
                    ],
                }
        return self.fallback.score_tokens(text, reference, image_path)

    def rank_score(
        self,
        text: str,
        reference: Optional[str] = None,
        image_path: Optional[str] = None,
        key_id: Optional[str] = None,
        text_key: Optional[str] = None,
    ) -> float:
        if key_id is not None:
            row = self._row_for(key_id)
            if row is not None:
                scores = row.get("scores") or {}
                lookup = text_key if text_key is not None else text
                if lookup in scores:
                    return float(scores[lookup])
        return self.fallback.rank_score(text, reference, image_path)


# ----------------------------------------------------------------------


REWRITE_PROMPT = (
    "You are rewriting a chest X-ray report so that it only contains findings "
    "observable from the CURRENT image. Remove all temporal and comparison "
    "statements (e.g. 'compared to prior', 'unchanged', 'interval change'). "
    "Keep all currently visible findings, devices, and their laterality. Use "
    "concise radiology style, lowercase, no headings.\n\n"
    "Original report:\n{reference}\n\n"
    "Rewritten current-observable report:"
)

SCORE_PROMPT_SUFFIX = "\n\nReport:\n{text}"

RANK_PROMPT = (
    "You are comparing two chest X-ray reports for the attached image. Reply "
    "with exactly 'A' or 'B' for the report that is more clinically accurate, "
    "complete, and free of hallucination or temporal claims.\n\n"
    "Report A:\n{candidate_a}\n\nReport B:\n{candidate_b}\n\n"
    "Better report:"
)


class LocalVLMTeacher(TeacherBase):
    """Image-grounded local multimodal teacher (separate environment only).

    Requires a modern ``transformers`` with vision-language support (e.g.
    MedGemma / Qwen2.5-VL class models). This repository pins 4.35.2, so run
    this class from its own environment and either use it directly in the
    offline CLIs there, or export JSONL and consume it via ``FileTeacher``.
    """

    name = "vlm"

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ) -> None:
        try:
            import torch  # noqa: F401
            import transformers
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "LocalVLMTeacher requires torch+transformers (>=4.5x with VLM "
                "support). Run it inside a separate teacher environment."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        self.dtype = dtype_map[dtype]

        self.processor = transformers.AutoProcessor.from_pretrained(model_name)
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=self.dtype, device_map=self.device
        )
        self.model.eval()

    def _load_image(self, image_path: Optional[str]):
        if image_path is None:
            return None
        from PIL import Image

        return Image.open(image_path).convert("RGB")

    def _chat(self, prompt: str, image=None) -> str:
        torch = self.torch
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-4),
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def rewrite_report(self, reference: str, image_path: Optional[str] = None) -> str:
        prompt = REWRITE_PROMPT.format(reference=str(reference))
        return self._chat(prompt, image=self._load_image(image_path))

    def score_tokens(
        self,
        text: str,
        reference: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Dict:
        """Teacher-forced log-probs of ``text`` conditioned on the image.

        The prompt asks the teacher to transcribe the report; the log-probs of
        the report tokens under that conditioning are the per-token scores.
        """
        torch = self.torch
        prompt = REWRITE_PROMPT.format(reference=str(reference or ""))
        suffix = SCORE_PROMPT_SUFFIX.format(text=str(text))
        image = self._load_image(image_path)

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": str(text)}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        # Identify the assistant suffix token span: teacher tokenisation of
        # the raw text, taken from the very end of the sequence.
        suffix_ids = self.processor(
            text=str(text), return_tensors="pt"
        )["input_ids"][0].to(self.model.device)
        total = inputs["input_ids"].shape[1]
        start = max(0, total - suffix_ids.shape[0])
        target_ids = inputs["input_ids"][0, start:]
        scores = log_probs[0, start - 1 : total - 1, :].gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)

        offsets = self._offsets_for(inputs, start, str(text))
        return {
            "teacher_token_logps": [float(v) for v in scores.tolist()],
            "teacher_token_offsets": offsets,
        }

    def _offsets_for(self, inputs, start: int, text: str) -> List[List[int]]:
        """Approximate char offsets for the scored teacher tokens.

        VLM processors do not uniformly expose offset mappings; we map each
        scored token to a slice of the text by decoding incrementally. Tokens
        that do not advance the decoded string (specials) are skipped by the
        aligner.
        """
        decoded_prefix = ""
        offsets: List[List[int]] = []
        base = 0
        for position in range(start, inputs["input_ids"].shape[1]):
            token_id = inputs["input_ids"][0, position]
            token_text = self.processor.decode(
                [token_id], skip_special_tokens=False
            )
            if not token_text.strip() and token_text not in (" ",):
                offsets.append([base, base])
                continue
            found = text.find(token_text, base)
            if found >= 0:
                offsets.append([found, found + len(token_text)])
                base = found + len(token_text)
            else:
                offsets.append([base, base])
        return offsets

    def rank_score(
        self,
        text: str,
        reference: Optional[str] = None,
        image_path: Optional[str] = None,
        peer: Optional[str] = None,
    ) -> float:
        """Pairwise win-rate proxy: P(text='A') - P(peer='B') via one prompt."""
        torch = self.torch
        prompt = RANK_PROMPT.format(candidate_a=str(text), candidate_b=str(peer or ""))
        image = self._load_image(image_path)
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        next_logits = logits[0, -1, :].float()
        log_probs = torch.log_softmax(next_logits, dim=-1)
        a_ids = [self.processor.encode("A", add_special_tokens=False)[-1]]
        b_ids = [self.processor.encode("B", add_special_tokens=False)[-1]]
        return float((log_probs[a_ids[0]] - log_probs[b_ids[0]]).item())


# ----------------------------------------------------------------------


def build_teacher(spec: str, **kwargs) -> TeacherBase:
    """Factory from a CLI spec string.

    - ``mock``
    - ``file:<path>``
    - ``vlm:<model_name>`` (e.g. ``vlm:google/medgemma-4b-it``)
    """
    if spec == "mock":
        return MockTeacher()
    if spec.startswith("file:"):
        return FileTeacher(spec[len("file:"):], mode=kwargs.get("mode", "auto"))
    if spec.startswith("vlm:"):
        return LocalVLMTeacher(spec[len("vlm:"):], **kwargs)
    raise ValueError(
        f"Unknown teacher spec: {spec!r}. Use 'mock', 'file:<path>', or "
        "'vlm:<model_name>'."
    )
