# MedGemma-RRG Alignment Plan

## Goal

Design MedGemma-RRG prediction, evaluation, and experiment outputs so they remain compatible with the original CvT2DistilGPT2 radiology report generation repository while adding the metadata needed for modern MedGemma inference, LoRA/QLoRA training, and future RadGraph/GREEN/CheXbert evaluation.

## 1. Canonical `predictions.jsonl` design

MedGemma-RRG should make JSONL the canonical prediction artifact:

```text
<run_dir>/predictions.jsonl
```

One line per evaluated sample:

```json
{
  "sample_id": "mimic_sample_0001",
  "dicom_id": "mimic_sample_0001",
  "study_id": null,
  "subject_id": null,
  "image_paths": ["p10/p10000032/s50414267/example.jpg"],
  "prediction": "the lungs are clear . no pleural effusion or pneumothorax .",
  "reference": "the lungs are clear . no pleural effusion or pneumothorax .",
  "findings_reference": null,
  "impression_reference": null,
  "split": "test",
  "epoch": null,
  "step": null,
  "metadata": {
    "dataset": "mimic_cxr_chen",
    "source_annotation_id": "mimic_sample_0001",
    "prompt_name": "default_rrg",
    "prompt": "Generate a radiology report for the provided chest X-ray.",
    "model_name": "google/medgemma-...",
    "checkpoint_path": null,
    "adapter_path": null,
    "generation_config": {
      "max_new_tokens": 128,
      "num_beams": 4,
      "temperature": null,
      "top_p": null
    }
  }
}
```

### Fields inherited from the original repository

| MedGemma field | Original source / mapping |
|---|---|
| `sample_id` | Original annotation `id`. |
| `dicom_id` | Original generated-report CSV uses `dicom_id` populated from annotation `id`; use the true DICOM ID if known, otherwise equal `sample_id`. |
| `image_paths` | Original annotation `image_path` list, optionally resolved to absolute paths. |
| `prediction` | Original generated report CSV column `report`. |
| `reference` | Original cleaned label from annotation `report`; metrics use this in memory as `batch['labels']`. |
| `split` | Original top-level annotation split and output filename prefix. |
| `epoch` | Original generated-report filename contains epoch. Use `null` for one-off inference. |

### Fields added for MedGemma

| Field | Why add it |
|---|---|
| `study_id` | Needed for MIMIC study-level grouping and RadGraph/GREEN reporting. |
| `subject_id` | Needed for traceability and patient-level grouping. |
| `findings_reference` | Future section-level evaluation/prompting. Original repo does not use this. |
| `impression_reference` | Future section-level evaluation/prompting. Original repo does not use this. |
| `step` | Useful for checkpoint/eval during training. |
| `metadata.dataset` | Avoids relying on directory names. |
| `metadata.prompt_name` / `metadata.prompt` | LLM-specific reproducibility. |
| `metadata.model_name` | MedGemma model identity. |
| `metadata.checkpoint_path` / `adapter_path` | Base, LoRA, QLoRA reproducibility. |
| `metadata.generation_config` | Captures decoding config absent from original CSV. |

## 2. Should MedGemma also output original-compatible CSV?

Yes.

The original repository's only always-on prediction artifact is:

```csv
report,dicom_id
```

MedGemma should write this file in addition to JSONL:

```text
<run_dir>/generated_reports/test_reports_epoch-none_<timestamp>.csv
```

or during training:

```text
<run_dir>/generated_reports/val_reports_epoch-<epoch>_<timestamp>.csv
<run_dir>/generated_reports/test_reports_epoch-<epoch>_<timestamp>.csv
```

Legacy-compatible CSV:

```csv
report,dicom_id
"the lungs are clear . no pleural effusion or pneumothorax .",mimic_sample_0001
```

Mapping:

- `report` = `prediction`
- `dicom_id` = `dicom_id` if available, else `sample_id`

This keeps compatibility with original report-inspection conventions, but it is not sufficient for standalone evaluation because it lacks references.

## 3. How to save prediction, reference, image paths, prompt, and generation config

Use three coordinated artifacts:

### A. Canonical JSONL

```text
<run_dir>/predictions.jsonl
```

Contains full record including prediction/reference/image paths/prompt/generation config.

### B. Rich CSV

```text
<run_dir>/predictions.csv
```

Recommended header:

```csv
sample_id,dicom_id,study_id,subject_id,split,image_paths,prediction,reference,findings_reference,impression_reference,prompt_name,model_name,generation_config_json
```

Example:

```csv
sample_id,dicom_id,study_id,subject_id,split,image_paths,prediction,reference,findings_reference,impression_reference,prompt_name,model_name,generation_config_json
mimic_sample_0001,mimic_sample_0001,,,test,"[\"p10/.../example.jpg\"]","generated text","reference text",,,default_rrg,google/medgemma,"{\"num_beams\":4,\"max_new_tokens\":128}"
```

### C. Legacy CSV

```text
<run_dir>/generated_reports/test_reports_epoch-none_<timestamp>.csv
```

Header exactly:

```csv
report,dicom_id
```

## 4. Base MedGemma inference output directory

Recommended:

```text
outputs/medgemma_rrg/<dataset>/<run_name>/
├── config.yaml
├── predictions.jsonl
├── predictions.csv
├── generated_reports/
│   └── test_reports_epoch-none_<timestamp>.csv
├── metrics/
│   ├── metrics.json
│   ├── metrics.csv
│   └── per_example_scores.csv
└── logs/
    └── inference.log
```

Example:

```text
outputs/medgemma_rrg/mimic_cxr_chen/base_medgemma_test_2026-06-25/
```

## 5. LoRA / QLoRA training output directory

Recommended:

```text
outputs/medgemma_rrg/<dataset>/<train_run_name>/
├── config.yaml
├── checkpoints/
│   ├── best/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── latest/
├── generated_reports/
│   ├── val_reports_epoch-<epoch>_<timestamp>.csv
│   └── test_reports_epoch-<epoch>_<timestamp>.csv
├── predictions_val_epoch-<epoch>.jsonl
├── predictions_test_epoch-<epoch>.jsonl
├── metrics/
│   ├── val_metrics_epoch-<epoch>.json
│   ├── test_metrics_epoch-<epoch>.json
│   └── metrics.csv
└── logs/
    ├── train.log
    └── tensorboard/
```

For QLoRA, adapter files should still be under `checkpoints/best/` and `checkpoints/latest/`; quantization config should be captured in `config.yaml` and/or `metadata`.

## 6. Evaluation script input

The MedGemma evaluator should primarily read:

```text
<run_dir>/predictions.jsonl
```

Required columns/fields for evaluation:

- `prediction`
- `reference`
- `sample_id` or `dicom_id`

Optional fields used for grouping/reporting:

- `split`
- `dataset`
- `study_id`
- `subject_id`
- `image_paths`

Do **not** make the evaluator depend on the legacy `report,dicom_id` CSV because it lacks references.

## 7. How to make MedGemma outputs consumable by original metrics

Original COCO wrapper expects:

```python
predictions: list[str]
labels: list[str] or list[list[str]]
ids: list[str]
```

Mapping from JSONL:

```python
predictions = [row["prediction"] for row in rows]
labels = [[row["reference"]] for row in rows]
ids = [row.get("dicom_id") or row["sample_id"] for row in rows]
```

Original CheXbert wrapper expects:

```python
y_hat: list[str]
y: list[str]
ids: list[str]
```

Mapping from JSONL:

```python
y_hat = [row["prediction"] for row in rows]
y = [row["reference"] for row in rows]
ids = [row.get("dicom_id") or row["sample_id"] for row in rows]
```

Original report logger compatibility:

```python
reports = [row["prediction"] for row in rows]
dicom_ids = [row.get("dicom_id") or row["sample_id"] for row in rows]
```

Then write legacy CSV `report,dicom_id`.

## 8. Supporting RadGraph / GREEN / CheXbert later

To support future metrics, include these fields now:

- `prediction`: full generated report.
- `reference`: full reference report.
- `findings_reference`: optional section-specific text.
- `impression_reference`: optional section-specific text.
- `sample_id`: stable row ID.
- `study_id`: stable study ID for study-level metrics.
- `dicom_id`: image/report ID for legacy compatibility.
- `image_paths`: traceability.
- `metadata.dataset`: dataset-specific metric behavior.

RadGraph/GREEN are not implemented in the original repository, so their metric fields should be optional/null until available.

## 9. Compatibility-critical fields to preserve

For original annotation compatibility:

```text
train / val / test
id
image_path
report
```

For original generated-report CSV compatibility:

```text
report
dicom_id
```

For original metrics comparability:

```text
chen_bleu_1
chen_bleu_2
chen_bleu_3
chen_bleu_4
chen_meteor
chen_rouge
chen_cider
chen_num_examples
ce_precision_macro
ce_recall_macro
ce_f1_macro
ce_precision_micro
ce_recall_micro
ce_f1_micro
ce_precision_example
ce_recall_example
ce_f1_example
ce_num_examples
```

## 10. New fields MedGemma should add

Minimum new prediction fields:

```text
sample_id
prediction
reference
image_paths
split
metadata.dataset
metadata.model_name
metadata.prompt_name
metadata.generation_config
```

Strongly recommended:

```text
study_id
subject_id
findings_reference
impression_reference
prompt
checkpoint_path
adapter_path
epoch
step
```

## 11. Recommended prediction JSONL

```json
{
  "sample_id": "...",
  "dicom_id": "...",
  "study_id": "...",
  "subject_id": "...",
  "image_paths": ["..."],
  "prediction": "...",
  "reference": "...",
  "findings_reference": "...",
  "impression_reference": "...",
  "split": "test",
  "epoch": null,
  "step": null,
  "metadata": {
    "dataset": "mimic_cxr_chen",
    "source_annotation_id": "...",
    "prompt_name": "...",
    "prompt": "...",
    "model_name": "...",
    "checkpoint_path": null,
    "adapter_path": null,
    "generation_config": {}
  }
}
```

## 12. Legacy-compatible CSV

Original source-backed format:

```csv
report,dicom_id
"generated report text",sample_or_dicom_id
```

MedGemma mapping:

- `report` = `prediction`
- `dicom_id` = true `dicom_id` if available, else original annotation `id` / `sample_id`

## 13. Metrics JSON

Use original metric names and add future metric placeholders:

```json
{
  "num_examples": 0,
  "chen_num_examples": 0,
  "chen_bleu_1": 0.0,
  "chen_bleu_2": 0.0,
  "chen_bleu_3": 0.0,
  "chen_bleu_4": 0.0,
  "chen_meteor": 0.0,
  "chen_rouge": 0.0,
  "chen_cider": 0.0,
  "ce_precision_macro": null,
  "ce_recall_macro": null,
  "ce_f1_macro": null,
  "ce_precision_micro": null,
  "ce_recall_micro": null,
  "ce_f1_micro": null,
  "ce_precision_example": null,
  "ce_recall_example": null,
  "ce_f1_example": null,
  "ce_num_examples": null,
  "radgraph_f1": null,
  "green": null
}
```

## 14. First implementation priorities for MedGemma-RRG

1. Original-compatible annotation loader.
2. Canonical JSONL prediction writer.
3. Legacy `report,dicom_id` CSV writer.
4. Evaluator that reads JSONL and computes `chen_*` metrics.
5. Optional CheXbert evaluator preserving `ce_*` fields.
6. Stable run directory layout with explicit `metrics/`, `generated_reports/`, and `logs/` directories.
