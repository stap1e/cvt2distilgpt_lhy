"""Fast CPU tests for the torch-free parts of tools/distillation.

These tests intentionally avoid importing torch/transformers so they can run
on any machine (the model-side GKD step is exercised through compileall and
GPU smoke runs on the training box).
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.distillation.current_observable import (
    CurrentObservableReportFilterV2,
)
from tools.distillation.gkd_pool import GKDPool, sequence_advantages, whiten
from tools.distillation.rollout_store import (
    group_by_id,
    mean,
    percentile,
    read_jsonl,
    write_jsonl,
)
from tools.distillation.run_recorder import (
    RunRecorder,
    collect_callback_metrics,
    is_recordable_metric,
)
from tools.distillation.teacher import MockTeacher, build_teacher
from tools.distillation.token_align import (
    align_teacher_to_student,
    spans_from_offsets,
)


# ----------------------------------------------------------------------
# Current-observable filter (Form A second pass)
# ----------------------------------------------------------------------


def test_filter_removes_temporal_sentences():
    filt = CurrentObservableReportFilterV2()
    report = (
        "Compared to the prior study, cardiomegaly is stable. "
        "Small left pleural effusion is present."
    )
    filtered, stats = filt.filter_report(report)
    assert "compared to" not in filtered.lower()
    assert "stable" not in filtered.lower()
    assert "pleural effusion" in filtered.lower()
    assert stats["removed"] == 1


def test_filter_rewrites_device_transitions():
    filt = CurrentObservableReportFilterV2()
    filtered, stats = filt.filter_report("The patient has been extubated.")
    assert "endotracheal tube is not present" in filtered.lower()
    assert stats["rewritten"] == 1


def test_filter_fallback_on_temporal_only_report():
    filt = CurrentObservableReportFilterV2(fallback="keep_original")
    report = "Interval change is noted since the previous study."
    filtered, stats = filt.filter_report(report)
    assert stats["fallback"] == 1
    assert filtered == report  # keep_original fallback keeps the raw text


def test_filter_detects_temporal_claims():
    filt = CurrentObservableReportFilterV2()
    assert filt.contains_temporal_claim("unchanged since prior exam")
    assert not filt.contains_temporal_claim("small pleural effusion on the left")


# ----------------------------------------------------------------------
# Cross-tokenizer alignment
# ----------------------------------------------------------------------


def test_alignment_by_char_overlap():
    student_spans = [(0, 3), (3, 7), (8, 13)]
    teacher_spans = [(0, 7), (8, 13)]  # teacher merges the first two tokens
    teacher_logps = [-0.5, -0.1]
    scores, coverage = align_teacher_to_student(
        student_spans, teacher_spans, teacher_logps
    )
    assert scores[0] == pytest.approx(-0.5)  # overlap with merged token
    assert scores[1] == pytest.approx(-0.5)
    assert scores[2] == pytest.approx(-0.1)
    assert coverage == pytest.approx(1.0)


def test_alignment_nearest_fallback_when_no_overlap():
    student_spans = [(0, 2), (5, 9)]
    teacher_spans = [(2, 5)]
    teacher_logps = [-1.0]
    scores, coverage = align_teacher_to_student(
        student_spans, teacher_spans, teacher_logps
    )
    # No strict overlap anywhere -> nearest fallback, coverage 0.
    assert scores == [-1.0, -1.0]
    assert coverage == 0.0


def test_alignment_empty_teacher_side():
    scores, coverage = align_teacher_to_student(
        [(0, 3)], [], [], unmatched=-4.0
    )
    assert scores == [-4.0]
    assert coverage == 0.0


def test_spans_from_offsets_drops_specials():
    spans = spans_from_offsets([[0, 0], [0, 4], [4, 5], [5, 5]])
    assert spans == [(0, 4), (4, 5)]


# ----------------------------------------------------------------------
# GKD pool + reward math
# ----------------------------------------------------------------------


def _scored_row(example_id, source, k, logps, teacher_mean=None, score=0.5):
    return {
        "id": example_id,
        "round": 1,
        "source": source,
        "k": k,
        "text": " ".join("tok" for _ in logps),
        "reference": "ref",
        "student_token_ids": list(range(len(logps))),
        "teacher_token_logps": list(logps),
        "teacher_mean_logp": (
            teacher_mean if teacher_mean is not None else sum(logps) / len(logps)
        ),
        "teacher_score": score,
        "alignment_coverage": 1.0,
    }


@pytest.fixture()
def scored_file(tmp_path):
    rows = [
        _scored_row("a", "greedy", 0, [-0.2, -0.2], teacher_mean=-0.2, score=0.9),
        _scored_row("a", "sample", 1, [-0.1, -2.0], teacher_mean=-1.05, score=0.6),
        _scored_row("a", "sample", 2, [-0.1, -0.1], teacher_mean=-0.1, score=0.8),
        _scored_row("b", "sample", 1, [-0.3, -0.3], teacher_mean=-0.3, score=0.4),
    ]
    path = tmp_path / "scored_round1.jsonl"
    write_jsonl(rows, str(path))
    return path


def test_pool_select_dense_rl(scored_file):
    pool = GKDPool.load(str(scored_file))
    selection = pool.select("dense_rl")
    assert [r["k"] for r in selection["a"]] == [1, 2]  # greedy excluded
    assert len(selection["b"]) == 1


def test_pool_select_best_of_k(scored_file):
    pool = GKDPool.load(str(scored_file))
    selection = pool.select("best_of_k")
    # For 'a' the best sample is k=2 (teacher_score 0.8 > 0.6).
    assert selection["a"][0]["k"] == 2
    assert selection["b"][0]["k"] == 1


def test_pool_greedy_scores_use_logp_scale(scored_file):
    pool = GKDPool.load(str(scored_file))
    greedy = pool.greedy_teacher_scores()
    # teacher_mean_logp (-0.2), not the ranking scalar (0.9).
    assert greedy["a"] == pytest.approx(-0.2)


def test_pool_rejects_stale_alignment(tmp_path):
    rows = [_scored_row("a", "sample", 1, [-0.1, -0.2])]
    rows[0]["student_token_ids"] = rows[0]["student_token_ids"][:1]
    path = tmp_path / "stale.jsonl"
    write_jsonl(rows, str(path))
    with pytest.raises(ValueError, match="alignment is stale"):
        GKDPool.load(str(path))


def test_whiten_is_zero_mean():
    values = [1.0, 2.0, 3.0, 6.0]
    whitened = whiten(values)
    assert sum(whitened) == pytest.approx(0.0, abs=1e-9)
    assert max(abs(v) for v in whitened) <= 3.0


def test_sequence_advantages_sequence_mean():
    adv = sequence_advantages([-1.0, -3.0], baseline="sequence_mean")
    assert sum(adv) == pytest.approx(0.0, abs=1e-6)


def test_sequence_advantages_greedy_baseline():
    adv = sequence_advantages(
        [-1.0, -3.0], baseline="greedy_score", baseline_value=-2.0,
        normalise="none",
    )
    assert adv[0] > 0 > adv[1]  # -1 beats the greedy baseline, -3 loses


def test_sequence_advantages_requires_baseline_value():
    with pytest.raises(ValueError, match="baseline_value"):
        sequence_advantages([-1.0], baseline="greedy_score", baseline_value=None)


def test_sequence_advantages_clip():
    adv = sequence_advantages(
        [-10.0, 10.0, 0.0], baseline="sequence_mean",
        clip=2.0, normalise="none",
    )
    assert max(abs(v) for v in adv) <= 2.0


# ----------------------------------------------------------------------
# Run recorder
# ----------------------------------------------------------------------


def test_run_recorder_filters_compute_metrics(tmp_path):
    recorder = RunRecorder(str(tmp_path), "unit", meta={"train_mode": "gkd"})
    recorder.record_epoch_metrics(
        0,
        "train",
        {
            "train_ce_loss": 1.5,
            "train_gkd_loss": -0.2,
            "train_epoch_time": 123.0,  # must be excluded
            "gpu_memory_reserved": 20.0,  # must be excluded
        },
    )
    rows = read_jsonl(str(tmp_path / "new_lab_run_record.jsonl"))
    epoch_rows = [r for r in rows if r["event"] == "epoch_metrics"]
    assert epoch_rows[0]["metrics"] == {
        "train_ce_loss": 1.5,
        "train_gkd_loss": -0.2,
    }
    markdown = (tmp_path / "new_lab_run_record.md").read_text(encoding="utf-8")
    assert "train_gkd_loss" in markdown
    assert "epoch_time" not in markdown


def test_collect_callback_metrics_whitelist():
    collected = collect_callback_metrics(
        {
            "val_ce_f1_macro": 0.4,
            "val_chen_cider": 0.35,
            "train_loss": 1.0,
            "step_time": 0.1,
            "some_unprefixed": 1.0,
        }
    )
    assert collected == {
        "val_ce_f1_macro": 0.4,
        "val_chen_cider": 0.35,
        "train_loss": 1.0,
    }


def test_is_recordable_metric_rules():
    assert is_recordable_metric("train_ce_loss")
    assert is_recordable_metric("val_chen_cider")
    assert not is_recordable_metric("train_time")
    assert not is_recordable_metric("gpu_util")
    assert not is_recordable_metric("ce_loss")  # unprefixed


# ----------------------------------------------------------------------
# Teachers
# ----------------------------------------------------------------------


def test_mock_teacher_rank_in_unit_interval():
    teacher = MockTeacher()
    score = teacher.rank_score(
        "the lungs are clear . no acute cardiopulmonary abnormality .",
        reference="the lungs are clear . no acute cardiopulmonary abnormality .",
    )
    assert 0.0 <= score <= 1.0


def test_mock_teacher_penalises_hallucinated_clinical_tokens():
    teacher = MockTeacher()
    scored = teacher.score_tokens(
        "lungs clear effusion", reference="lungs are clear"
    )
    # 'effusion' is clinical and absent from the reference -> strong penalty.
    assert scored["teacher_token_logps"][-1] == pytest.approx(
        teacher.off_ref_clinical_logp
    )


def test_mock_teacher_rewrite_is_identity():
    teacher = MockTeacher()
    assert teacher.rewrite_report("hello world") == "hello world"


def test_file_teacher_falls_back_to_mock(tmp_path):
    from tools.distillation.teacher import FileTeacher

    teacher_file = tmp_path / "teacher.jsonl"
    write_jsonl([{"id": "a", "rewrite": "teacher rewrite"}], str(teacher_file))
    teacher = FileTeacher(str(teacher_file))
    assert teacher.rewrite_report("original", key_id="a") == "teacher rewrite"
    assert teacher.rewrite_report("original", key_id="missing") == "original"
    assert teacher.fallback_hits == 1


def test_build_teacher_mock_factory():
    assert isinstance(build_teacher("mock"), MockTeacher)


# ----------------------------------------------------------------------
# rollout_store helpers
# ----------------------------------------------------------------------


def test_jsonl_roundtrip_and_grouping(tmp_path):
    rows = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}, {"id": "a", "text": "z"}]
    path = tmp_path / "rows.jsonl"
    write_jsonl(rows, str(path))
    loaded = read_jsonl(str(path))
    assert loaded == rows
    assert set(group_by_id(loaded).keys()) == {"a", "b"}
    assert len(group_by_id(loaded)["a"]) == 2


def test_mean_and_percentile():
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert mean([]) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert percentile([], 0.5) is None
