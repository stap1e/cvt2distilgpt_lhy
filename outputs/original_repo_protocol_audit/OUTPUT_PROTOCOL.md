# Prediction / Generated Report Output Protocol

## Scope and sources

Read-only audit of generated report and prediction/result saving code in:

- `cvt2distilgpt2_mimic_cxr_chen.py`
- `cvt2distilgpt2_iu_x_ray_chen.py`
- `tools/metrics/report_logger.py`
- `tools/metrics/coco.py`
- `tools/metrics/chexbert.py`
- `stages.py`
- `README.md`

## 1. Where generated reports are saved

Generated reports are saved by `tools.metrics.report_logger.ReportLogger`.

Model constructors create:

```python
self.val_report_logger = ReportLogger(exp_dir=self.exp_dir_trial, split='val_reports')
self.test_report_logger = ReportLogger(exp_dir=self.exp_dir_trial, split='test_reports')
```

`ReportLogger` saves into:

```text
<exp_dir_trial>/generated_reports/
```

README examples:

```text
experiment/test_mimic_cxr_chen_cvt2distilgpt2/trial_0/generated_reports/test_reports_epoch-0_16-05-2023_10-20-48.csv
experiment/test_iu_x_ray_chen_cvt2distilgpt2/trial_0/generated_reports/test_reports_epoch-0_16-05-2023_12-46-42.csv
```

Current YAML configs use `/data/lhy_data/Cvt2distillgpt2/results_*` roots rather than README's `experiment/...` examples. The `generated_reports/` subdirectory convention is source-backed.

## 2. Save format

The always-on generated report output is **CSV**.

No always-on JSON, JSONL, TXT, or pickle prediction output was found.

Optional metric classes can save additional CSV files if constructor flags are enabled, but the model constructors do not enable those flags.

## 3. Generated report record fields

`ReportLogger.update(reports, dicom_ids)` appends dicts with exactly:

```python
{
    'report': generated_report_text,
    'dicom_id': id_from_batch
}
```

Therefore the generated report CSV columns are:

```csv
report,dicom_id
```

Important naming detail:

- `report` means **generated prediction**, not reference report.
- `dicom_id` is populated from `batch['id']`; it may be a sample/study ID depending on annotation contents.

Fields not saved by `ReportLogger`:

- reference / ground truth
- image path
- split as a separate column
- epoch as a separate column
- step
- generation config
- checkpoint path
- model name
- prompt
- dataset name

## 4. Does it save reference / ground truth?

The always-on generated report CSV does **not** save reference text.

References are passed in-memory to metrics:

```python
self.test_chexbert_metrics.update(generated, batch['labels'], ids=batch['id'])
self.test_coco_metrics.update(generated, [[i] for i in batch['labels']], ids=batch['id'])
```

The optional COCO save path can write both prediction and label if `COCOCaptionMetrics(save=True, exp_dir=...)` is used, but current model constructors instantiate `COCOCaptionMetrics(...)` without `save=True`.

Optional COCO `predictions.csv` header:

```csv
prediction,label,id
```

## 5. Does it save image path?

No generated report output saves image paths.

Dataset returns image path only for MIMIC as `image_filepaths`; validation/test steps do not log it. IU dataset does not return image paths.

## 6. Does it save split / epoch / step / decoding config?

- Split: encoded in filename prefix: `val_reports` or `test_reports`.
- Epoch: encoded in filename: `epoch-<epoch>`.
- Timestamp: encoded in filename.
- Step: not saved.
- Decoding config: not saved in prediction files.
  - Test uses `num_test_beams` from config.
  - Validation uses `num_beams=1` hard-coded.
  - `decoder_max_len` is used in generation.
- Experiment name/path: implicit in the directory path, not in rows.

## 7. Deduplication behavior

`ReportLogger.log()` creates a DataFrame and drops duplicates by `dicom_id`:

```python
df = pd.DataFrame(self.reports).drop_duplicates(subset='dicom_id')
```

Default pandas behavior keeps the first occurrence.

Metric classes also deduplicate in some paths:

- CheXbert: drops duplicates by `ids` before scoring.
- COCO: builds dicts keyed by `id`, so duplicate IDs overwrite earlier values while constructing `predictions` and `labels`.

## 8. File naming rules

Generated report CSV:

```text
<exp_dir_trial>/generated_reports/<split>_epoch-<epoch>_<DD-MM-YYYY_HH-MM-SS>.csv
```

Examples:

```text
.../generated_reports/val_reports_epoch-0_16-05-2023_10-20-48.csv
.../generated_reports/test_reports_epoch-0_16-05-2023_10-20-48.csv
```

Where:

- `<split>` is the `ReportLogger` constructor argument, currently `val_reports` or `test_reports`.
- `<epoch>` is `self.current_epoch` passed at epoch end.
- Timestamp uses `time.strftime("%d-%m-%Y_%H-%M-%S")`.

Optional metric files:

- COCO `predictions.csv` under `exp_dir` when `save=True`; fixed filename, overwritten.
- COCO `individual_scores.csv` under `exp_dir` when `save_individual_scores=True`; fixed filename, overwritten.
- COCO `bootstrapped_scores.csv` under `exp_dir` when `save_bootstrapped_scores=True`; appended.
- CheXbert `ce_class_metrics.csv`, `ce_class_metrics_2.csv`, etc. under `exp_dir` when `save_class_scores=True`.
- CheXbert `chexbert_outputs_<timestamp>.csv` under `exp_dir` when `save_outputs=True`, semicolon-delimited.

## 9. Output directory construction

Primary generated report directory:

```text
exp_dir_trial
└── generated_reports
    ├── val_reports_epoch-<epoch>_<timestamp>.csv
    └── test_reports_epoch-<epoch>_<timestamp>.csv
```

`exp_dir_trial` is passed into the model from `dlhpcstarter`; it is not constructed in the visible model code. Based on README examples and config conventions it likely includes:

```text
<exp_dir>/<task>/<config_name>/trial_<trial>/
```

However, the exact construction is external to this repository.

## 10. Checkpoint directory vs generated report directory

There are two checkpoint-related concepts:

1. **Model zoo/checkpoint assets** controlled by `ckpt_zoo_dir`:
   - CvT checkpoint.
   - DistilGPT2 local Hugging Face files.
   - CheXbert checkpoint.

2. **Experiment checkpoints** under the experiment trial directory or explicit `test_ckpt_path`:
   - selected by `get_test_ckpt_path(args.exp_dir_trial, args.monitor, args.monitor_mode, args.test_epoch, args.test_ckpt_path)`.
   - `write_test_ckpt_path(ckpt_path, args.exp_dir_trial)` records the selected test checkpoint path.

Generated reports are under the trial directory, not under `ckpt_zoo_dir`.

## 11. Intermediate result files

Always-on intermediate/generated files found:

- generated report CSV files under `generated_reports/`.
- test checkpoint path bookkeeping via external `write_test_ckpt_path(...)` in `stages.py`; exact filename is defined by `dlhpcstarter`, not visible here.

Optional files if metric save flags are enabled:

- `predictions.csv`
- `individual_scores.csv`
- `bootstrapped_scores.csv`
- `ce_class_metrics*.csv`
- `chexbert_outputs_<timestamp>.csv`

No JSONL/JSON intermediate prediction file was found.

## 12. Per-example score files

Supported but not enabled by current model constructors:

- COCO per-example scores via `COCOCaptionMetrics(save_individual_scores=True)`:
  - saves `<exp_dir>/individual_scores.csv`
  - DataFrame columns are metric names such as `chen_bleu_1`, `chen_bleu_2`, `chen_bleu_3`, `chen_bleu_4`, `chen_meteor`, `chen_rouge`, `chen_cider`, `chen_spice` depending on enabled metrics.
  - The file as written does not explicitly include IDs unless the DataFrame index is interpreted externally.

CheXbert optional output is per-example-ish and semicolon-delimited:

- `<exp_dir>/chexbert_outputs_<timestamp>.csv`
- columns in table:
  - `chexbert_y_hat`
  - `chexbert_y`
  - `y_hat`
  - `y`
  - `ids`

## 13. Final metrics summary files

No final `metrics.json` or `metrics.csv` summary file is always written by this repository's model code.

Metric dicts are returned by metric classes and logged with Lightning:

```python
self.log_dict({f'test_{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)
```

Actual persisted logs/checkpoint monitor files are delegated to PyTorch Lightning / `dlhpcstarter` logger and callback configuration, which is external to this repository.

## 14. How outputs are read by later evaluation scripts

No standalone evaluation script that reads `generated_reports/*.csv` was found in this repository.

Metrics are computed during validation/test directly from in-memory predictions/references. The generated report CSV is primarily an artifact for inspection or downstream manual use.

The optional COCO and CheXbert save paths produce CSVs, but there is no in-repo script shown that reads them back.

## 15. Original repository prediction CSV example

Generated by `ReportLogger`:

```csv
report,dicom_id
"the lungs are clear . no pleural effusion or pneumothorax .",mimic_sample_0001
```

Column semantics:

- `report`: generated report/prediction.
- `dicom_id`: value from annotation `id` after batch collation.

## 16. Optional COCO prediction CSV example

Only if `COCOCaptionMetrics(save=True, exp_dir=...)` is enabled:

```csv
prediction,label,id
"the lungs are clear . no pleural effusion .","the lungs are clear . no pleural effusion or pneumothorax .",mimic_sample_0001
```

This is not enabled in current model constructors.

## 17. Recommended MedGemma-compatible prediction format

The MedGemma-RRG repository should preserve the legacy CSV while adopting a richer JSONL as the canonical output.

### Canonical JSONL record

```json
{
  "sample_id": "mimic_sample_0001",
  "dicom_id": "mimic_sample_0001",
  "study_id": null,
  "subject_id": null,
  "image_paths": ["p10/p10000032/s50414267/example.jpg"],
  "prediction": "the lungs are clear . no pleural effusion or pneumothorax .",
  "reference": "the lungs are clear . no pleural effusion or pneumothorax .",
  "split": "test",
  "epoch": null,
  "step": null,
  "metadata": {
    "dataset": "mimic_cxr_chen",
    "prompt_name": "default_rrg",
    "model_name": "google/medgemma-...",
    "checkpoint_path": null,
    "generation_config": {
      "max_new_tokens": 128,
      "num_beams": 4
    }
  }
}
```

### Field provenance

Original-compatible fields:

- `sample_id`: maps from original `id`.
- `dicom_id`: for legacy compatibility, can equal original `id` unless true DICOM IDs are available.
- `image_paths`: maps from original `image_path` list or absolute `image_file_path` list.
- `prediction`: maps from original generated `report` column.
- `reference`: maps from original `label` / cleaned annotation report. Raw `report` can be kept separately in metadata if needed.
- `split`: source split, originally encoded by top-level JSON key and output filename.
- `epoch`: originally encoded in filename during Lightning validation/test.

MedGemma-added fields:

- `study_id`
- `subject_id`
- `step`
- `metadata.dataset`
- `metadata.prompt_name`
- `metadata.model_name`
- `metadata.checkpoint_path`
- `metadata.generation_config`

### Legacy-compatible CSV for original metrics/logging style

Because the original always-on generated report artifact uses `report,dicom_id`, MedGemma should also write:

```csv
report,dicom_id
"the lungs are clear . no pleural effusion or pneumothorax .",mimic_sample_0001
```

Recommended filename:

```text
<run_dir>/generated_reports/test_reports_epoch-0_<timestamp>.csv
```

or, for non-training inference:

```text
<run_dir>/generated_reports/test_reports_epoch-none_<timestamp>.csv
```

### Rich CSV for direct metric consumption

For easier evaluation, also consider a richer CSV:

```csv
sample_id,dicom_id,study_id,subject_id,split,image_paths,prediction,reference
mimic_sample_0001,mimic_sample_0001,,,test,"[\"p10/.../example.jpg\"]","generated text","reference text"
```

This richer CSV is not original-compatible but is useful for MedGemma evaluators.
