# Reuse Recommendations for MedGemma-RRG

## Scope

This document classifies original-repository components by how suitable they are for reuse in a new MedGemma-RRG repository.

## Can be reused directly or nearly directly

### 1. Report cleaning logic

- Original files:
  - `tools/dataset/mimic_cxr_chen_tokenizer.py`
  - `tools/dataset/iu_x_ray_chen_tokenizer.py`
- Role:
  - Normalize Chen/R2Gen report text.
  - Lowercase, remove punctuation, normalize sentence boundaries, and produce `" . "`-separated reports.
- How to reuse in MedGemma-RRG:
  - Extract cleaner functions into a dataset preprocessing module, e.g. `medgemma_rrg/data/report_cleaning.py`.
  - Use them to produce legacy-compatible `reference` strings for COCO/CheXbert comparison.
  - Keep raw report text separately if prompt construction should use raw style.
- Field-name changes:
  - None required if input still uses `report`.
  - If using `findings`/`impression`, add explicit logic rather than overloading `report`.
- Risks:
  - The cleaner is tailored to Chen/R2Gen labels and may remove punctuation/section structure useful for LLM prompting.
  - Vocabulary thresholding in `TokenizerChen` can convert rare words to `<unk>` before decoding labels; for MedGemma, reuse cleaning but avoid thresholded-vocab truncation unless reproducing old metrics exactly.

### 2. COCO caption metric wrapper

- Original file:
  - `tools/metrics/coco.py`
- Role:
  - Computes BLEU-1/2/3/4, METEOR, ROUGE, CIDEr, and optionally SPICE through `pycocoevalcap`.
- How to reuse in MedGemma-RRG:
  - Feed `predictions: list[str]`, `labels: list[list[str]]` or `list[str]`, and `ids: list[str]` from `predictions.jsonl`.
  - Preserve original `chen_*` metric field names for comparability.
- Field-name changes:
  - Input JSONL can use `prediction`, `reference`, and `sample_id`; evaluator maps these to `predictions`, `labels`, `ids`.
- Risks:
  - METEOR/SPICE may require Java and extra resources.
  - Duplicate IDs silently overwrite entries.
  - Original `individual_scores.csv` lacks explicit IDs; improve if adapting.

### 3. CheXbert metrics wrapper

- Original files:
  - `tools/metrics/chexbert.py`
  - `tools/chexbert.py`
- Role:
  - Runs CheXbert on generated/reference reports and computes CE precision/recall/F1 macro/micro/example metrics.
- How to reuse in MedGemma-RRG:
  - Reuse as evaluator module when CheXbert checkpoint is available.
  - Input can come from `prediction` and `reference` fields in JSONL.
  - Preserve `ce_*` field names.
- Field-name changes:
  - None for metric outputs.
  - Map MedGemma `sample_id` or `dicom_id` to `ids`.
- Risks:
  - Requires local CheXbert checkpoint and compatible dependencies.
  - Uses only positive CheXbert labels (`== 1`) for scoring.
  - Loading CheXbert inside each `compute()` can be expensive.

## Can be adapted and reused

### 1. Dataset annotation parser

- Original files:
  - `cvt2distilgpt2_mimic_cxr_chen.py::setup()` and `format_examples()`
  - `cvt2distilgpt2_iu_x_ray_chen.py::setup()` and `format_examples()`
  - `tools/dataset/mimc_cxr_chen.py`
  - `tools/dataset/iu_x_ray_chen.py`
- Role:
  - Reads `{train, val, test}` annotation JSON.
  - Expects `id`, `image_path`, `report` per example.
  - Resolves image paths and prepares labels.
- How to reuse in MedGemma-RRG:
  - Implement a framework-neutral parser that reads the same JSON protocol.
  - Preserve support for `id`, `image_path`, `report`.
  - Add optional support for `subject_id`, `study_id`, `dicom_id`, `findings`, `impression`.
- Field-name changes:
  - Keep `id`, `image_path`, `report` for legacy input compatibility.
  - Add normalized output fields `sample_id`, `image_paths`, `reference`.
- Risks:
  - MIMIC old code uses only first image; MedGemma may need multiple images.
  - IU old code assumes exactly/at least two images in list order.
  - No view filtering exists; if needed, add explicit metadata handling.

### 2. `ReportLogger`

- Original file:
  - `tools/metrics/report_logger.py`
- Role:
  - Writes generated predictions as CSV with columns `report,dicom_id` under `<exp_dir>/generated_reports`.
- How to reuse in MedGemma-RRG:
  - Keep a legacy-compatible writer that emits exactly `report,dicom_id`.
  - Build a new rich writer for JSONL/CSV with prediction, reference, image paths, prompt, and generation config.
- Field-name changes:
  - Legacy mode: no changes.
  - Rich mode: use `prediction` instead of `report` to avoid ambiguity.
- Risks:
  - Original logger drops duplicate `dicom_id` and silently truncates if `reports` and `dicom_ids` lengths differ due to `zip`.
  - It does not save references; not enough for standalone evaluation.

### 3. Metric calculator orchestration

- Original files:
  - `cvt2distilgpt2_mimic_cxr_chen.py::validation_step/test_step/on_*_epoch_end`
  - `tools/metrics/coco.py`
  - `tools/metrics/chexbert.py`
- Role:
  - Updates metrics in memory during validation/test and logs summary dicts through Lightning.
- How to reuse in MedGemma-RRG:
  - Move metric computation into a standalone evaluator that reads `predictions.jsonl`.
  - Keep metric class wrappers but decouple them from training loop.
- Field-name changes:
  - Read `prediction`/`reference` from JSONL.
  - Output original metric names in `metrics.json` and `metrics.csv`.
- Risks:
  - Original code does not evaluate from saved generated report CSV, because that CSV lacks references.

### 4. Config field ideas

- Original files:
  - `config/*.yaml`
- Role:
  - Control paths, module class, hyperparameters, checkpoint selection, decoding length/beam count.
- How to reuse in MedGemma-RRG:
  - Use clearer config fields:
    - `dataset_dir`
    - `annotation_path`
    - `image_root`
    - `output_dir`
    - `run_name`
    - `prediction_path`
    - `metrics_dir`
    - `generation.max_new_tokens`
    - `generation.num_beams`
  - Preserve old `decoder_max_len` and `num_test_beams` aliases if migrating configs.
- Field-name changes:
  - Prefer `output_dir`/`run_dir` over ambiguous `exp_dir`.
- Risks:
  - `exp_dir_trial` is externally produced by `dlhpcstarter`; avoid depending on that hidden convention.

## Not recommended to reuse

### 1. Old model class and decoder tokenization path

- Original files:
  - `cvt2distilgpt2_mimic_cxr_chen.py`
  - `cvt2distilgpt2_iu_x_ray_chen.py`
- Role:
  - CvT encoder + DistilGPT2 decoder LightningModules.
- Why not reuse:
  - MedGemma has a different model architecture and likely Hugging Face multimodal generation path.
  - The old decoder uses GPT2-specific BOS/PAD handling and EncoderDecoderModel wrapper hacks.
- How to replace:
  - Implement MedGemma dataset-to-prompt-to-generate pipeline separately.
  - Reuse only dataset parsing, cleaning, logging, metrics concepts.
- Risk:
  - Copying old model code would add unnecessary technical debt.

### 2. Old training loop / `stages.py` as-is

- Original file:
  - `stages.py`
- Role:
  - `dlhpcstarter` stage function for train/test orchestration.
- Why not reuse:
  - Depends on `dlhpcstarter` and hidden `exp_dir_trial` conventions.
  - Current `trainer.fit(...)` lines are commented out.
  - MedGemma LoRA/QLoRA training likely uses different trainer stack.
- How to replace:
  - Use explicit scripts such as `train_lora.py`, `infer.py`, and `evaluate.py` or a modern CLI.
  - Keep output directory conventions from this audit.
- Risk:
  - Direct reuse may produce confusing partial behavior.

### 3. `NaturalLanguage` base class as-is

- Original file:
  - `tools/metrics/natural_language.py`
- Role:
  - Intended generic metric base.
- Why not reuse:
  - Base `compute()` references undefined states `self.predictions` and `self.labels` and does not return `self.scores(df)`.
- How to replace:
  - Use a simpler evaluator class or fix before reuse.
- Risk:
  - Direct reuse can fail if subclass does not override `compute()`.

### 4. Old image handling assumptions as-is

- Original files:
  - `tools/dataset/mimc_cxr_chen.py`
  - `tools/dataset/iu_x_ray_chen.py`
  - `tools/multi_image.py`
- Role:
  - MIMIC loads first image only; IU loads first two images.
- Why not reuse directly:
  - MedGemma may need explicit multi-image support and view metadata.
  - Silent first-image behavior can lose relevant views.
- How to adapt:
  - Preserve original behavior under `legacy_image_policy`.
  - Add explicit policies: `first`, `all`, `frontal_only`, `frontal_lateral_pair`.
- Risk:
  - Output comparability may change if image selection changes.

## Highest-value reusable design patterns

1. **Annotation JSON split protocol**
   - `{train, val, test}` with `id`, `image_path`, `report`.
2. **Legacy generated report CSV**
   - `generated_reports/<split>_reports_epoch-<epoch>_<timestamp>.csv`
   - columns `report,dicom_id`.
3. **Metric field names**
   - `chen_*` for COCO metrics.
   - `ce_*` for CheXbert metrics.
4. **In-memory metric input shape**
   - predictions: `list[str]`
   - references: `list[str]` or `list[list[str]]`
   - ids: `list[str]`
5. **Trial directory structure**
   - Keep `generated_reports/`, add `metrics/`, `predictions.jsonl`, and logs/checkpoints explicitly.

## Recommended migration strategy

1. Implement a MedGemma dataset loader that accepts original annotation JSON unchanged.
2. Produce canonical `predictions.jsonl` with rich fields.
3. Also produce legacy CSV `generated_reports/test_reports_epoch-none_<timestamp>.csv` with `report,dicom_id`.
4. Implement evaluator that reads `predictions.jsonl`, computes COCO/CheXbert, and writes `metrics.json` / `metrics.csv`.
5. Keep original metric names for comparability.
6. Add optional fields for RadGraph/GREEN later without changing the legacy CSV.
