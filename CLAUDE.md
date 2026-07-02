# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repository implements CvT2DistilGPT2 for chest X-ray report generation. The model pairs a CvT-21 image encoder with a DistilGPT2 decoder configured for cross-attention, using PyTorch Lightning and `dlhpcstarter` for train/test orchestration.

There are two task-specific LightningModules:

- `cvt2distilgpt2_mimic_cxr_chen.py` defines `CvT2DistilGPT2MIMICXRChen` for MIMIC-CXR with Chen/R2Gen annotations.
- `cvt2distilgpt2_iu_x_ray_chen.py` defines `CvT2DistilGPT2IUXRayChen`, which subclasses the MIMIC module and adapts paths/datasets for IU X-Ray two-image studies via `tools/multi_image.py`.

`stages.py` is the `dlhpcstarter` entrypoint. It loads YAML config into args, imports the configured model class from `module` / `definition`, creates the Lightning Trainer, optionally trains, and tests by resolving/loading a checkpoint before calling `trainer.test(model)`.

## Environment and dependencies

Install dependencies from `requirements.txt`:

```bash
python -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt --no-cache-dir
```

Main pinned dependencies include `dlhpcstarter==0.1.2`, `transformers==4.35.2`, `timm==0.4.12`, `yacs==0.1.8`, and COCO caption metric packages.

## Common commands

Run syntax/compile smoke check:

```bash
python -m compileall tools stages.py cvt2distilgpt2_mimic_cxr_chen.py cvt2distilgpt2_iu_x_ray_chen.py infer.py
```

Check for tests:

```bash
python -m pytest --collect-only -q
```

This repository currently does not include test files, so pytest may report `no tests collected`.

Run MIMIC-CXR testing:

```bash
dlhpcstarter -t mimic_cxr_chen -c config/test_mimic_cxr_chen_cvt2distilgpt2.yaml --stages_module stages --test
```

Run IU X-Ray testing:

```bash
dlhpcstarter -t iu_x_ray_chen -c config/test_iu_x_ray_chen_cvt2distilgpt2.yaml --stages_module stages --test
```

Run MIMIC-CXR training:

```bash
dlhpcstarter -t mimic_cxr_baseline -c config/train_mimic_cxr_chen_cvt2distilgpt2.yaml --stages_module stages --train
```

Run IU X-Ray training:

```bash
dlhpcstarter -t iu_x_ray_chen -c config/train_iu_x_ray_chen_cvt2distilgpt2.yaml --stages_module stages --train
```

Run standalone inference without `dlhpcstarter`:

```bash
python infer.py --config config/test_iu_x_ray_chen_cvt2distilgpt2.yaml --ckpt-path checkpoints/iu_x_ray_chen/cvt_21_to_distilgpt2/epoch=10-val_chen_cider=0.475024.ckpt --output-path outputs/iu_predictions.jsonl --device cuda:0 --max-samples 10
```

There is no configured lint/format command in this checkout.

## Data and checkpoint expectations

Config files define these important roots:

- `exp_dir`: experiment/output base directory.
- `dataset_dir`: dataset base directory.
- `ckpt_zoo_dir`: checkpoint/model-zoo base directory.
- `test_ckpt_path`: checkpoint to load for test configs.

The current source expects dataset files under paths derived from `dataset_dir`:

- MIMIC labels: `<dataset_dir>/mimic_cxr_chen/annotation.json`
- MIMIC images: `<dataset_dir>/mimic_cxr_chen/mimic_cxr_jpg/...`
- IU labels: `<dataset_dir>/iu_x-ray_chen/annotation.json`
- IU images: `<dataset_dir>/iu_x-ray_chen/images/...`

This differs from parts of the README, which refers to `annotations.json` and a separate `dataset/mimic_cxr_jpg` tree. Follow the code/config when debugging runtime path errors.

Model resources are loaded locally, not downloaded at runtime:

- CvT checkpoint is loaded by `tools/cvt.py` from `<ckpt_zoo_dir>/microsoft/CvT/CvT-21-384x384-IN-22k.pth`.
- MIMIC DistilGPT2 files are loaded from `<ckpt_zoo_dir>/distilbert/distilgpt2`.
- IU DistilGPT2 files are loaded from `<ckpt_zoo_dir>/distilgpt2`.
- CheXbert metrics load `<ckpt_zoo_dir>/stanford/chexbert/chexbert.pth`.

## Architecture notes

The high-level runtime flow is:

```text
YAML config
  -> dlhpcstarter calls stages.stages(args)
  -> stages.py imports args.module / args.definition
  -> LightningModule builds datasets, tokenizer, encoder, projection, decoder, metrics
  -> setup(stage) loads annotation.json splits and formats examples
  -> DataLoader yields images, labels, and decoder token tensors for training
  -> training_step uses teacher forcing and cross-entropy loss
  -> validation/test generate reports with model.generate()
  -> ReportLogger writes generated report CSVs; COCO and CheXbert metrics are logged
```

Key implementation details:

- `tools/cvt.py` wraps the vendored CvT implementation in `tools/ext/cvt`. When used as an encoder it returns flattened spatial features as `last_hidden_state`.
- `tools/encoder_projection.py` permutes CvT output and linearly projects encoder hidden size 384 to DistilGPT2 hidden size 768.
- The decoder is a Hugging Face `EncoderDecoderModel` assembled with a dummy encoder and a GPT2 LM head decoder. The real encoder outputs are passed through `encoder_outputs` during forward/generation.
- GPT2 tokenizers add `[BOS]` and `[PAD]`; training tokenization in `tools/dataset/dataset.py` manually shifts decoder inputs and labels.
- `tools/dataset/mimc_cxr_chen.py` loads one image per example. `tools/dataset/iu_x_ray_chen.py` loads two images and stacks them; `CvT2DistilGPT2IUXRayChen.encoder_forward()` flattens/restores multi-image features.
- Validation uses greedy generation (`num_beams=1`); test uses `num_test_beams` from config.
- Generated reports are written under `<exp_dir_trial>/generated_reports/` by `tools/metrics/report_logger.py`.

## Config caveats

- The README's IU training command appears to repeat the MIMIC command. Use `config/train_iu_x_ray_chen_cvt2distilgpt2.yaml` with task `iu_x_ray_chen` for IU work.
- Some config files use absolute local paths under `/data/lhy_data/rg`; update `exp_dir`, `dataset_dir`, `ckpt_zoo_dir`, and `test_ckpt_path` before running elsewhere.
- `DataLoader` calls pass `prefetch_factor`; PyTorch requires `num_workers > 0` for this. `infer.py` guards against `num_workers=0`, but the LightningModule dataloaders do not.
