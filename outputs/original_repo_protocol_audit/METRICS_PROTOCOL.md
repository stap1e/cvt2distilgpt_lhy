# Metrics Input / Output Protocol

## Scope and sources

Read-only audit of metric implementations and metric usage in:

- `cvt2distilgpt2_mimic_cxr_chen.py`
- `cvt2distilgpt2_iu_x_ray_chen.py`
- `tools/metrics/coco.py`
- `tools/metrics/chexbert.py`
- `tools/metrics/natural_language.py`
- `tools/metrics/report_logger.py`
- `tools/chexbert.py`

## 1. Implemented metrics

| Metric family | Implemented? | File | Output field names |
|---|---:|---|---|
| BLEU-1/2/3/4 | Yes | `tools/metrics/coco.py` | `chen_bleu_1`, `chen_bleu_2`, `chen_bleu_3`, `chen_bleu_4` |
| BLEU-4 | Yes, as part of BLEU | `tools/metrics/coco.py` | `chen_bleu_4` |
| METEOR | Yes | `tools/metrics/coco.py` | `chen_meteor` |
| ROUGE-L / ROUGE | Yes via pycocoevalcap Rouge | `tools/metrics/coco.py` | `chen_rouge` |
| CIDEr | Yes | `tools/metrics/coco.py` | `chen_cider` |
| SPICE | Supported by wrapper but not enabled in model constructors | `tools/metrics/coco.py` | `chen_spice` |
| CheXbert / CE metrics | Yes | `tools/metrics/chexbert.py`, `tools/chexbert.py` | `ce_precision_*`, `ce_recall_*`, `ce_f1_*`, `ce_num_examples` |
| RadGraph | No active implementation found | N/A | N/A |
| GREEN | No active implementation found | N/A | N/A |

Model usage:

- Validation COCO metrics: `COCOCaptionMetrics(metrics=["bleu", "cider", "rouge"])`
- Test COCO metrics: `COCOCaptionMetrics(metrics=["bleu", "cider", "meteor", "rouge"])`
- CheXbert metrics: both validation and test.
- SPICE is available in the wrapper but not requested by current model constructors.

## 2. COCO caption metrics protocol

Implementation: `tools/metrics/coco.py::COCOCaptionMetrics`

### Constructor

```python
COCOCaptionMetrics(
    metrics=["bleu", "cider", "meteor", "rouge", "spice"],
    save=False,
    save_individual_scores=False,
    save_bootstrapped_scores=False,
    exp_dir=None,
    dist_sync_on_step=False,
)
```

### In-memory input to `update()`

```python
update(predictions, labels, ids)
```

Expected:

- `predictions`: list of strings.
- `labels`: either list of strings or list of list-of-strings. Current model passes `[[i] for i in batch['labels']]`.
- `ids`: list/sequence of identifiers.

The current model calls:

```python
self.test_coco_metrics.update(generated, [[i] for i in batch['labels']], ids=batch['id'])
```

### Internal scoring format

Before pycocoevalcap scoring, the wrapper creates dictionaries:

```python
predictions[id] = [prediction_text]
labels[id] = [reference_text]
```

It normalizes periods with:

```python
re.sub(' +', ' ', text.replace('.', ' .'))
```

If a label entry is a list, it uses only the first reference:

```python
report_str = k[0] if isinstance(k, list) else k
```

### Output from `compute()` / `score()`

Returns a Python dict, for enabled metrics:

```python
{
  "chen_bleu_1": float,
  "chen_bleu_2": float,
  "chen_bleu_3": float,
  "chen_bleu_4": float,
  "chen_meteor": float,
  "chen_rouge": float,
  "chen_cider": float,
  "chen_spice": float,
  "chen_num_examples": int
}
```

Actual keys depend on the requested `metrics` list.

### Saved files, if enabled

Current model constructors do **not** enable these save flags. If enabled:

1. `save=True`

```text
<exp_dir>/predictions.csv
```

Header:

```csv
prediction,label,id
```

2. `save_individual_scores=True`

```text
<exp_dir>/individual_scores.csv
```

Columns are per-example metric score arrays, e.g.:

```text
chen_bleu_1, chen_bleu_2, chen_bleu_3, chen_bleu_4, chen_meteor, chen_rouge, chen_cider, chen_spice
```

The saved file does not explicitly include `id` as a column.

3. `save_bootstrapped_scores=True`

```text
<exp_dir>/bootstrapped_scores.csv
```

Appends accumulated metric dict rows. Header is written only if the file does not exist.

## 3. CheXbert / CE metrics protocol

Implementation:

- `tools/metrics/chexbert.py::CheXbertMetrics`
- `tools/chexbert.py::CheXbert`

### Constructor

```python
CheXbertMetrics(
    ckpt_dir,
    bert_path,
    checkpoint_path,
    mbatch_size=16,
    save_class_scores=False,
    save_outputs=False,
    exp_dir=None,
)
```

Current model passes:

```python
CheXbertMetrics(
    bert_path='bert-base-uncased',
    checkpoint_path='stanford/chexbert/chexbert.pth',
    ckpt_dir=self.ckpt_zoo_dir,
    mbatch_size=self.mbatch_size,
    exp_dir=self.exp_dir_trial,
)
```

Save flags are left at default `False`.

### In-memory input

`CheXbertMetrics` inherits `NaturalLanguage.update(y_hat, y, ids)`, so input is:

```python
update(y_hat, y, ids)
```

Expected:

- `y_hat`: list of generated report strings.
- `y`: list of reference report strings.
- `ids`: list/sequence of identifiers.

Current model calls:

```python
self.test_chexbert_metrics.update(generated, batch['labels'], ids=batch['id'])
```

### Internal scoring format

- Runs CheXbert on generated reports and reference reports.
- Produces multi-label class predictions for 14 conditions:
  - `enlarged_cardiomediastinum`
  - `cardiomegaly`
  - `lung_opacity`
  - `lung_lesion`
  - `edema`
  - `consolidation`
  - `pneumonia`
  - `atelectasis`
  - `pneumothorax`
  - `pleural_effusion`
  - `pleural_other`
  - `fracture`
  - `support_devices`
  - `no_finding`
- Class value convention in comments:
  - `0 = blank/not mentioned`
  - `1 = positive`
  - `2 = negative`
  - `3 = uncertain`
- Metrics are computed on positive labels only: `(df == 1)`.

### Output from `compute()`

Returns a Python dict:

```python
{
  "ce_precision_macro": float,
  "ce_recall_macro": float,
  "ce_f1_macro": float,
  "ce_precision_micro": float,
  "ce_recall_micro": float,
  "ce_f1_micro": float,
  "ce_precision_example": float,
  "ce_recall_example": float,
  "ce_f1_example": float,
  "ce_num_examples": float
}
```

### Saved files, if enabled

Current model constructors do **not** enable these flags.

1. `save_class_scores=True`

Path:

```text
<exp_dir>/ce_class_metrics.csv
<exp_dir>/ce_class_metrics_2.csv
...
```

Uses `enumerated_save_path()` to avoid overwriting.

Columns are per-class metric fields, e.g.:

```text
ce_precision_cardiomegaly, ce_recall_cardiomegaly, ce_f1_cardiomegaly, ...
```

2. `save_outputs=True`

Path:

```text
<exp_dir>/chexbert_outputs_<DD-MM-YYYY_HH-MM-SS>.csv
```

Delimiter:

```text
;
```

Columns:

```text
chexbert_y_hat;chexbert_y;y_hat;y;ids
```

Where:

- `chexbert_y_hat`: CheXbert condition labels for generated reports.
- `chexbert_y`: CheXbert condition labels for reference reports.
- `y_hat`: generated report string.
- `y`: reference report string.
- `ids`: sample identifier.

## 4. NaturalLanguage base class note

`tools/metrics/natural_language.py` defines a generic `NaturalLanguage` metric with a `pairs` state and `update(y_hat, y, ids)`.

However, its base `compute()` references `self.predictions` and `self.labels`, which are not defined in the base class, and it does not return `self.scores(df)`. In this repository, `CheXbertMetrics` overrides `compute()`, so this is mostly latent. The base class should not be directly reused without fixing.

## 5. Metrics saving location

Always-on metric values:

- Returned as dicts from metric classes.
- Logged through Lightning using `self.log_dict({f'val_{k}': v ...})` or `self.log_dict({f'test_{k}': v ...})`.
- The exact persistent logger files are governed by Lightning / `dlhpcstarter`, not visible in this repository.

Optional CSV saves:

- COCO optional files: under `exp_dir` passed to the metric. In the current model, `exp_dir=self.exp_dir_trial` is not passed for COCO metrics, so enabling `save=True` would also require passing `exp_dir`.
- CheXbert optional files: under `self.exp_dir_trial` because model passes `exp_dir=self.exp_dir_trial`.

## 6. Does evaluation depend on generated report files?

No. Current validation/test metrics are computed directly in memory during Lightning epoch-end.

Flow:

```text
generated strings + reference labels + ids
  -> metric.update(...)
  -> metric.compute()
  -> Lightning self.log_dict(...)
```

Generated report CSV files are not read back by in-repo evaluation code.

## 7. Duplicate handling

- COCO: stores predictions/labels in dicts keyed by `id`; duplicate IDs overwrite earlier entries during dict construction.
- CheXbert: drops duplicate rows by `ids` before computing metrics.
- ReportLogger: drops duplicate generated reports by `dicom_id` before saving.

For MedGemma, IDs must be unique per evaluated sample to avoid silent overwrites/deduplication.

## 8. Components that can migrate to MedGemma-RRG

### Directly reusable with minimal changes

- `tools/metrics/coco.py::COCOCaptionMetrics`
  - Input is generic lists of prediction/reference strings and IDs.
  - Keep `chen_*` field names if comparing to original repository.
  - Requires pycocoevalcap and Java-backed metrics for METEOR/SPICE depending on environment.

- `tools/metrics/chexbert.py::CheXbertMetrics`
  - Input is generic generated/reference strings and IDs.
  - Requires CheXbert checkpoint and dependencies.
  - Keep `ce_*` field names for compatibility.

### Adapt before reuse

- `tools/metrics/report_logger.py::ReportLogger`
  - Useful legacy CSV writer.
  - Add reference, sample ID, image paths, split, generation config in MedGemma-specific richer outputs.
  - Keep a legacy mode that writes only `report,dicom_id`.

- `tools/metrics/natural_language.py`
  - Do not reuse base `compute()` without fixing the undefined-state issue.

## 9. Field names to preserve for compatibility

Original metric names:

- `chen_bleu_1`
- `chen_bleu_2`
- `chen_bleu_3`
- `chen_bleu_4`
- `chen_meteor`
- `chen_rouge`
- `chen_cider`
- `chen_spice` if used
- `chen_num_examples`
- `ce_precision_macro`
- `ce_recall_macro`
- `ce_f1_macro`
- `ce_precision_micro`
- `ce_recall_micro`
- `ce_f1_micro`
- `ce_precision_example`
- `ce_recall_example`
- `ce_f1_example`
- `ce_num_examples`

Original generated report CSV fields:

- `report`
- `dicom_id`

Optional COCO save fields:

- `prediction`
- `label`
- `id`

## 10. Recommended unified MedGemma-RRG evaluator protocol

MedGemma-RRG evaluator should read a canonical prediction file:

```text
<run_dir>/predictions.jsonl
```

Each row should include at least:

```json
{
  "sample_id": "...",
  "dicom_id": "...",
  "study_id": "...",
  "subject_id": "...",
  "image_paths": ["..."],
  "prediction": "...",
  "reference": "...",
  "split": "test",
  "metadata": {
    "dataset": "mimic_cxr_chen",
    "model_name": "...",
    "prompt_name": "...",
    "generation_config": {}
  }
}
```

Then produce:

```text
<run_dir>/metrics/metrics.json
<run_dir>/metrics/metrics.csv
<run_dir>/metrics/per_example_scores.csv        # optional
<run_dir>/metrics/chexbert_outputs.csv          # optional
```

### Recommended `metrics.json`

Use original field names for direct comparability:

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

### Recommended `metrics.csv`

One row per evaluation run:

```csv
run_id,split,num_examples,chen_bleu_1,chen_bleu_2,chen_bleu_3,chen_bleu_4,chen_meteor,chen_rouge,chen_cider,ce_f1_macro,ce_f1_micro,ce_f1_example,radgraph_f1,green
```

### Recommended `per_example_scores.csv`

```csv
sample_id,dicom_id,split,chen_bleu_1,chen_bleu_2,chen_bleu_3,chen_bleu_4,chen_meteor,chen_rouge,chen_cider,chexbert_match_f1,radgraph_f1,green_score
```

Notes:

- Original COCO wrapper can compute per-example arrays but does not save IDs in `individual_scores.csv`; MedGemma should add IDs.
- RadGraph and GREEN are not in the original repository, so their fields should be optional/null until implemented.
