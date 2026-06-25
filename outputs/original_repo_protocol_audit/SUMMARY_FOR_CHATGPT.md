# Summary for ChatGPT: Original Repository Protocol Audit

This is a compact summary of the original CvT2DistilGPT2 radiology report generation repository protocols for reuse in a new MedGemma-RRG repository.

## 1. Minimal data sample format

The original repository expects an annotation JSON file with top-level splits:

```json
{
  "train": [
    {
      "id": "sample_0001",
      "image_path": ["relative/path/to/image.jpg"],
      "report": "The lungs are clear. No pleural effusion or pneumothorax."
    }
  ],
  "val": [],
  "test": []
}
```

Required fields per sample:

- `id`: sample identifier. It is later used as metric ID and saved as `dicom_id` in generated-report CSVs.
- `image_path`: list of relative image paths.
- `report`: raw report text.

Dataset-specific paths:

- MIMIC-CXR source expects:
  - annotation: `<dataset_dir>/mimic_cxr_chen/annotation.json`
  - images: `<dataset_dir>/mimic_cxr_chen/mimic_cxr_jpg/files`
- IU X-Ray source expects:
  - annotation: `<dataset_dir>/iu_x-ray_chen/annotation.json`
  - images: `<dataset_dir>/iu_x-ray_chen/images`

Important mismatch: README says `annotations.json`, but source expects singular `annotation.json`.

Image behavior:

- MIMIC-CXR loads only `image_path[0]`; extra paths are ignored.
- IU X-Ray loads `image_path[0]` and `image_path[1]` and stacks them as two views.
- No explicit view filtering exists.

Report target behavior:

- The model does not use raw `report` directly.
- It cleans the report with a Chen/R2Gen-style tokenizer, truncates to 60 Chen-token IDs, then decodes back to normalized text.
- No separate `findings` or `impression` fields are consumed by source code.

## 2. Generated report output format

The always-on generated report output is CSV, written by `tools/metrics/report_logger.py::ReportLogger`.

Path:

```text
<exp_dir_trial>/generated_reports/<split>_epoch-<epoch>_<DD-MM-YYYY_HH-MM-SS>.csv
```

Examples:

```text
generated_reports/val_reports_epoch-0_16-05-2023_10-20-48.csv
generated_reports/test_reports_epoch-0_16-05-2023_10-20-48.csv
```

CSV header:

```csv
report,dicom_id
```

Example:

```csv
report,dicom_id
"the lungs are clear . no pleural effusion or pneumothorax .",sample_0001
```

Field meanings:

- `report`: generated prediction text, not reference text.
- `dicom_id`: copied from annotation `id` via `batch['id']`.

Not saved in this legacy CSV:

- reference / ground truth
- image path
- split column
- step
- generation config
- model/prompt/checkpoint metadata

Deduplication:

- `ReportLogger` drops duplicate rows by `dicom_id` before saving.

## 3. Metrics input/output format

Metrics are computed in memory during validation/test. The generated-report CSV is not read back by an in-repo evaluator.

### COCO caption metrics

File: `tools/metrics/coco.py`

Input:

```python
predictions: list[str]
labels: list[str] or list[list[str]]
ids: list[str]
```

Model call:

```python
COCOCaptionMetrics.update(generated, [[label] for label in batch['labels']], ids=batch['id'])
```

Output dict keys:

```text
chen_bleu_1
chen_bleu_2
chen_bleu_3
chen_bleu_4
chen_meteor
chen_rouge
chen_cider
chen_spice      # supported but not enabled by current model constructors
chen_num_examples
```

Current validation uses BLEU, CIDEr, ROUGE. Current test uses BLEU, CIDEr, METEOR, ROUGE.

Optional COCO files, only if flags are enabled:

- `<exp_dir>/predictions.csv` with `prediction,label,id`
- `<exp_dir>/individual_scores.csv`
- `<exp_dir>/bootstrapped_scores.csv`

These optional saves are not enabled in the current model constructors.

### CheXbert metrics

Files:

- `tools/metrics/chexbert.py`
- `tools/chexbert.py`

Input:

```python
y_hat: list[str]      # generated reports
y: list[str]          # reference reports
ids: list[str]
```

Output dict keys:

```text
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

Optional CheXbert files, only if flags are enabled:

- `<exp_dir>/ce_class_metrics.csv`, `<exp_dir>/ce_class_metrics_2.csv`, ...
- `<exp_dir>/chexbert_outputs_<timestamp>.csv` using semicolon delimiter with columns `chexbert_y_hat;chexbert_y;y_hat;y;ids`

These optional saves are not enabled by default in the current model constructors.

RadGraph and GREEN are not implemented in the original repository.

## 4. Experiment directory structure

The source-backed generated-report structure is:

```text
<exp_dir_trial>/
└── generated_reports/
    ├── val_reports_epoch-<epoch>_<timestamp>.csv
    └── test_reports_epoch-<epoch>_<timestamp>.csv
```

Checkpoint examples in configs imply Lightning-style checkpoint names:

```text
epoch=42-step=521289-val_chen_cider=0.431418.ckpt
epoch=11-step=2076-val_chen_cider=0.444897.ckpt
```

Checkpoint selection uses:

- `monitor: val_chen_cider`
- `monitor_mode: max`
- `test_ckpt_path` in test configs, if supplied

`exp_dir_trial` construction is external to this repo via `dlhpcstarter`. Based on examples, it is likely under:

```text
<exp_dir>/<task>/<config_name>/trial_<trial>/
```

but the exact rule is not defined in this repository.

No explicit `metrics.json`, `metrics.csv`, config snapshot, TensorBoard logger, WandB logger, or CSV logger is directly implemented in this repository; logging is delegated to Lightning / `dlhpcstarter`.

## 5. Fields MedGemma-RRG should preserve

For input annotation compatibility:

```text
train / val / test
id
image_path
report
```

For legacy generated-report CSV compatibility:

```text
report
dicom_id
```

For metric-name compatibility:

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

## 6. Recommended MedGemma-RRG output protocol

Canonical prediction artifact:

```text
<run_dir>/predictions.jsonl
```

Recommended JSONL row:

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

Also write legacy-compatible CSV:

```text
<run_dir>/generated_reports/test_reports_epoch-none_<timestamp>.csv
```

```csv
report,dicom_id
"generated report text",sample_or_dicom_id
```

Recommended metrics outputs:

```text
<run_dir>/metrics/metrics.json
<run_dir>/metrics/metrics.csv
<run_dir>/metrics/per_example_scores.csv      # optional
```

Recommended `metrics.json` fields should preserve original names:

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

## 7. New repository first-stage implementation files

Suggested initial MedGemma-RRG files:

1. `data/chen_annotation.py`
   - Load original `{train,val,test}` JSON.
   - Normalize `id`, `image_path`, `report` to `sample_id`, `image_paths`, `reference`.

2. `data/report_cleaning.py`
   - Port/adapt Chen report cleaning from original tokenizers.

3. `inference/generate.py` or `scripts/infer.py`
   - Run MedGemma inference and write `predictions.jsonl`.

4. `utils/prediction_io.py`
   - JSONL writer, rich CSV writer, legacy `report,dicom_id` CSV writer.

5. `evaluation/coco_metrics.py`
   - Port/adapt `COCOCaptionMetrics` or wrap pycocoevalcap.

6. `evaluation/chexbert_metrics.py`
   - Port/adapt CheXbert metrics if checkpoint/dependencies are available.

7. `scripts/evaluate.py`
   - Read `predictions.jsonl`, write `metrics.json`, `metrics.csv`, optional per-example scores.

8. `configs/*.yaml`
   - Explicit fields for annotation path, image root, output directory, model name, prompt name, generation config.

## 8. Best original files to reuse

Most useful:

- `tools/metrics/coco.py`
  - Reuse metric names and list/dict scoring protocol.
- `tools/metrics/chexbert.py`
  - Reuse CE metric definitions and field names.
- `tools/chexbert.py`
  - Reuse CheXbert model wrapper if dependencies match.
- `tools/metrics/report_logger.py`
  - Reuse as a legacy CSV writer pattern, but extend elsewhere for JSONL.
- `tools/dataset/mimic_cxr_chen_tokenizer.py`
  - Reuse cleaning logic, not necessarily thresholded vocabulary behavior.
- `tools/dataset/iu_x_ray_chen_tokenizer.py`
  - Same for IU cleaning.

Avoid direct reuse:

- `cvt2distilgpt2_mimic_cxr_chen.py`
- `cvt2distilgpt2_iu_x_ray_chen.py`
- `stages.py`
- `tools/metrics/natural_language.py` as-is

## 9. Biggest risks

1. **`id` is overloaded**
   - Original code uses `id` as metrics ID and generated CSV `dicom_id`. It may not be a true DICOM ID.

2. **Generated report CSV lacks references**
   - Cannot perform standalone evaluation from `report,dicom_id` alone.

3. **README/source annotation filename mismatch**
   - README says `annotations.json`; source expects `annotation.json`.

4. **MIMIC image handling is first-image-only**
   - Extra images in `image_path` are ignored.

5. **IU image handling assumes first two paths**
   - No explicit view labels/filtering.

6. **Metrics are in-memory in original training loop**
   - A new MedGemma evaluator must read JSONL or rich CSV, not the original generated-report CSV.

7. **External `dlhpcstarter` hides experiment/log details**
   - Do not rely on hidden `exp_dir_trial` construction in the new repo.

8. **CheXbert requires local checkpoint/dependencies**
   - Keep it optional and record nulls when unavailable.
