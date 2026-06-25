# Entrypoints and Execution Flow

## Scope and sources

Read-only audit based on `README.md`, `stages.py`, `cvt2distilgpt2_mimic_cxr_chen.py`, `cvt2distilgpt2_iu_x_ray_chen.py`, and all YAML files under `config/`.

## 1. Training entrypoints

The repository does not expose a standalone `train.py`. Training is launched through `dlhpcstarter` with this repository's `stages.py` as the stage module.

README training commands:

```bash
dlhpcstarter -t mimic_cxr -c config/train_mimic_cxr_chen_cvt2distilgpt2.yaml --stages_module stages --train --test
```

```bash
dlhpcstarter -t mimic_cxr -c config/train_mimic_cxr_chen_cvt2distilgpt2.yaml --stages_module stages --train --test
```

Notes:

- The second README command is under the IU X-Ray section but still uses the MIMIC config/task name. This appears to be a README copy-paste error. The actual IU training config is `config/train_iu_x_ray_chen_cvt2distilgpt2.yaml`, whose commented example uses `-t iu_x_ray_test`.
- The train configs define `module` and `definition`, which tell `stages.py` which LightningModule class to import.

Training-related files:

- `stages.py`: generic train/test orchestration.
- `cvt2distilgpt2_mimic_cxr_chen.py`: `CvT2DistilGPT2MIMICXRChen` LightningModule.
- `cvt2distilgpt2_iu_x_ray_chen.py`: `CvT2DistilGPT2IUXRayChen`, subclassing the MIMIC module and overriding IU-specific dataset/multi-image behavior.
- `config/train_mimic_cxr_chen_cvt2distilgpt2.yaml`
- `config/train_iu_x_ray_chen_cvt2distilgpt2.yaml`

## 2. Evaluation / testing entrypoints

Testing is also launched through `dlhpcstarter` + `stages.py` with `--test`.

README testing commands:

```bash
dlhpcstarter -t mimic_cxr_chen -c config/test_mimic_cxr_chen_cvt2distilgpt2.yaml --stages_module stages --test
```

```bash
dlhpcstarter -t iu_x_ray_chen -c config/test_iu_x_ray_chen_cvt2distilgpt2.yaml --stages_module stages --test
```

Testing config files:

- `config/test_mimic_cxr_chen_cvt2distilgpt2.yaml`
- `config/test_iu_x_ray_chen_cvt2distilgpt2.yaml`

Testing loads a checkpoint selected by `get_test_ckpt_path(...)` or specified in `test_ckpt_path`, then calls `trainer.test(model)`.

## 3. Inference / generation entrypoints

There is no separate inference-only script. Generation happens inside Lightning validation/test steps:

- MIMIC validation: `validation_step()` calls `generate(1, batch['encoder_images'])` for greedy generation.
- MIMIC test: `test_step()` calls `generate(self.num_test_beams, batch['encoder_images'])` for beam search.
- IU X-Ray inherits the validation/test generation and logging logic from the MIMIC class; it overrides encoder handling for two images.

Generated text is decoded via `self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)` and logged by `ReportLogger`.

## 4. Purpose of `stages.py`

`stages.py` defines a single `stages(args: Namespace)` function. It is the `dlhpcstarter` stage hook.

Responsibilities:

1. Set `args.warm_start_modules = False` initially.
2. Load the YAML config and update `args` via `load_config_and_update_args(args)`.
3. Seed Lightning with either `args.seed` or `args.trial`.
4. Dynamically import the task model class from `args.module` and `args.definition`.
5. Build a PyTorch Lightning trainer via `trainer_instance(**vars(args))`.
6. If `args.train`: instantiate or warm-start the model. In the current source, `trainer.fit(...)` is commented out, so this checkout does not actually train unless those lines are restored.
7. If `args.test`: resolve a checkpoint path, write it to the experiment trial directory, load the model with `load_from_checkpoint(..., strict=False)`, then call `trainer.test(model)`.

Important source detail: in the current `stages.py`, the lines that would resume and call `trainer.fit(model, ckpt_path=ckpt_path)` are commented out. This affects whether README training commands actually train in this checkout.

## 5. Purpose of `cvt2distilgpt2_*` scripts

### `cvt2distilgpt2_mimic_cxr_chen.py`

Defines `CvT2DistilGPT2MIMICXRChen`, a LightningModule for MIMIC-CXR report generation with Chen/R2Gen annotations.

Main responsibilities:

- Construct MIMIC annotation path:
  - `<dataset_dir>/mimic_cxr_chen/annotation.json`
- Construct MIMIC image root:
  - `<dataset_dir>/mimic_cxr_chen/mimic_cxr_jpg/files`
- Build a Chen tokenizer from train reports.
- Instantiate COCO caption metrics and CheXbert metrics.
- Instantiate `ReportLogger` for validation and test generated reports.
- Build CvT encoder, projection layer, DistilGPT2 decoder, and GPT2 tokenizer.
- Define image transforms.
- Load train/val/test examples in `setup()`.
- Convert annotation fields:
  - `image_path` -> `image_file_path`
  - `report` -> `label`
  - relative image paths -> absolute paths under image root
  - label is cleaned/truncated with `TokenizerChen` and decoded back into normalized text.
- Provide dataloaders.
- Train with teacher forcing.
- Validate/test by generation, report logging, CheXbert metrics, and COCO metrics.

### `cvt2distilgpt2_iu_x_ray_chen.py`

Defines `CvT2DistilGPT2IUXRayChen`, a subclass of the MIMIC class for IU X-Ray.

IU-specific changes:

- Annotation path:
  - `<dataset_dir>/iu_x-ray_chen/annotation.json`
- Image root:
  - `<dataset_dir>/iu_x-ray_chen/images`
- Uses `tools.dataset.iu_x_ray_chen.TaskSubset`.
- Uses `tools.dataset.iu_x_ray_chen_tokenizer.TokenizerChen`.
- Assumes two images per example in the dataset class.
- Adds `MultiImageInput` and `MultiImageOutput` to flatten two images for the encoder and concatenate encoded image features per example.
- Overrides `encoder_forward()` for multi-image input.

## 6. Runtime config requirements

Every runtime config needs at least:

- `exp_dir`: base experiment/output directory.
- `dataset_dir`: base dataset directory.
- `ckpt_zoo_dir`: base checkpoint/model-zoo directory.
- `module`: Python module name for the LightningModule.
- `definition`: class name in that module.
- training hyperparameters: `encoder_lr`, `decoder_lr`, `mbatch_size`, `decoder_max_len`, etc.
- trainer controls: `devices`, `num_nodes`, `num_workers`, `strategy`, `precision`, `max_epochs` for training configs.
- metric/checkpoint selection: `monitor`, `monitor_mode`, and usually `test_ckpt_path` for test configs.
- generation control: `num_test_beams` and `decoder_max_len`.

`dlhpcstarter` appears to derive `exp_dir_trial` from `exp_dir`, task name/config name, and trial number. Source for that derivation is external to this repository.

## 7. Config fields controlling protocol-relevant behavior

| Concern | Field(s) | Observed values / behavior |
|---|---|---|
| Dataset base path | `dataset_dir` | MIMIC and IU modules append dataset-specific subpaths. |
| MIMIC annotation path | derived from `dataset_dir` | `<dataset_dir>/mimic_cxr_chen/annotation.json` in source. README says `annotations.json`, but source expects singular `annotation.json`. |
| MIMIC image root | derived from `dataset_dir` | `<dataset_dir>/mimic_cxr_chen/mimic_cxr_jpg/files`. README describes `dataset/mimic_cxr_jpg/...`; current source differs. |
| IU annotation path | derived from `dataset_dir` | `<dataset_dir>/iu_x-ray_chen/annotation.json`. README says `annotations.json`, but source expects `annotation.json`. |
| IU image root | derived from `dataset_dir` | `<dataset_dir>/iu_x-ray_chen/images`. |
| Experiment base directory | `exp_dir` | e.g. `/data/lhy_data/Cvt2distillgpt2/results_mimic_cxr`; external `dlhpcstarter` creates trial directory. |
| Experiment trial directory | `exp_dir_trial` | Passed into model constructor by `dlhpcstarter`; used for generated reports and CheXbert optional saves. |
| Checkpoint/model zoo directory | `ckpt_zoo_dir` | Used for CvT, DistilGPT2, CheXbert resources. |
| Test checkpoint | `test_ckpt_path` | Explicit absolute or relative `.ckpt` path in test configs. |
| Checkpoint selection metric | `monitor`, `monitor_mode` | e.g. `val_chen_cider`, `max`. |
| Generated reports save path | derived from `exp_dir_trial` | `<exp_dir_trial>/generated_reports/<split>_epoch-<epoch>_<timestamp>.csv`. |
| Metrics save path | mostly in Lightning logs/stdout | COCO/CheXbert scores are logged via `self.log_dict`; optional CSV save flags exist in metric classes but are not enabled in the model constructors. |
| Batch size | `mbatch_size` | Used by dataloaders and CheXbert mini-batch size. |
| Max generated/training length | `decoder_max_len` | Used in training tokenization and generation `max_length`. |
| Beam size | `num_test_beams` | Test generation only. Validation uses greedy search (`num_beams=1`). |
| Decoding parameters | `decoder_max_len`, `num_test_beams` | `bos_token_id`, `eos_token_id`, `pad_token_id`, `use_cache=True` are hard-coded. No temperature/top-p controls. |

## 8. Main flow from dataset to metrics

```text
YAML config
  -> dlhpcstarter parses task/config/trial and calls stages.stages(args)
  -> stages.py loads config and imports args.module / args.definition
  -> LightningModule constructor
       -> builds dataset paths from dataset_dir
       -> builds tokenizer / encoder / decoder / metrics / ReportLogger
  -> Lightning setup(stage)
       -> json.load(annotation.json)
       -> split examples: train / val / test
       -> format_examples()
            image_path -> absolute image_file_path list
            report -> cleaned/truncated label string
       -> TaskSubset examples
  -> DataLoader
       -> TaskSubset.__getitem__()
            loads image(s)
            returns id, encoder_images, labels, optional decoder ids/masks for training
  -> training_step()
       -> model.forward(images, decoder_input_ids, decoder_attention_mask)
       -> cross entropy against label_ids
       -> Lightning train_loss logging
  -> validation_step() / test_step()
       -> model.generate(images)
       -> tokenizer.batch_decode(...)
       -> ReportLogger.update(generated, dicom_ids=batch['id'])
       -> CheXbertMetrics.update(generated, labels, ids=batch['id'])
       -> COCOCaptionMetrics.update(generated, [[label]], ids=batch['id'])
  -> on_validation_epoch_end() / on_test_epoch_end()
       -> ReportLogger.compute(epoch)
            saves CSV under generated_reports/
       -> CheXbertMetrics.compute()
       -> COCOCaptionMetrics.compute()
       -> self.log_dict(val_* or test_* scores)
  -> output files and Lightning logs
```

## 9. Important inconsistencies / unknowns

- README says annotation files are `annotations.json`; source expects `annotation.json`.
- README training command for IU appears to repeat the MIMIC command.
- Current `stages.py` comments out the actual `trainer.fit(...)` call; training flow is therefore incomplete in this checkout.
- `exp_dir_trial` construction and checkpoint callback/log directory details are owned by external `dlhpcstarter`, not visible in this repository.
