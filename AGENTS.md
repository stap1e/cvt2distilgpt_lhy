# Repository Guidelines

## Project Structure & Module Organization

Root-level `cvt2distilgpt2_*.py` files define the PyTorch Lightning models for MIMIC-CXR, IU X-Ray, and visual-grounded variants. `stages.py` and `stages_visual_grounded*.py` are `dlhpcstarter` train/test entry points; `infer.py` provides standalone checkpoint inference. Shared datasets, metrics, reward logic, and model components live under `tools/`; `tools/ext/cvt/` is vendored CvT code and should receive only targeted upstream-compatible edits. YAML experiment definitions live in `config/`. `dataset/` and `checkpoints/` are local placeholders, `docs/` holds figures, and `experiment/` contains example generated reports.

## Setup, Test, and Development Commands

There is no packaging build step. The documented environment reuses system-installed PyTorch/Lightning and adds pinned packages:

```bash
python -m venv --system-site-packages venv
python -m pip install --upgrade -r requirements.txt --no-cache-dir
```

Use these common commands from the repository root:

```bash
python -m compileall tools stages.py cvt2distilgpt2_mimic_cxr_chen.py cvt2distilgpt2_iu_x_ray_chen.py infer.py
dlhpcstarter -t mimic_cxr_baseline -c config/train_mimic_cxr_chen_cvt2distilgpt2.yaml --stages_module stages --train
dlhpcstarter -t mimic_cxr_chen -c config/test_mimic_cxr_chen_cvt2distilgpt2.yaml --stages_module stages --test
python infer.py --config config/test_iu_x_ray_chen_cvt2distilgpt2.yaml --ckpt-path <model.ckpt> --output-path outputs/predictions.jsonl --device cuda:0 --max-samples 10
```

For IU X-Ray or grounded models, select the matching task, config, and `stages` module.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8 conventions. Name modules, functions, arguments, and YAML keys with `snake_case`; use `PascalCase` for model and dataset classes and `UPPER_CASE` for constants. Keep imports grouped as standard library, third-party, then local. No formatter or linter is configured, so avoid unrelated formatting churn and follow the surrounding file.

## Testing Guidelines

The repository currently has no unit-test directory or coverage threshold. Treat `compileall` as the minimum smoke check, then run the relevant config-driven test or a small `infer.py --max-samples` job. New isolated logic should include fast CPU tests under `tests/`, named `test_*.py`, using `pytest`; mock large datasets and checkpoints rather than adding them to the repository.

## Commit & Pull Request Guidelines

History favors short, single-line summaries such as `Fix from_pretrained call for CheXbert.` Use an imperative summary for one logical change and mention the affected model or config when useful. Pull requests should explain purpose, list changed configs, report verification commands and key metrics, link related issues, and include sample output when generated reports change.

## Data & Configuration Safety

Several YAML files contain machine-specific roots. Update `dataset_dir`, `ckpt_zoo_dir`, `exp_dir`, and checkpoint paths locally. Do not commit protected clinical data, model weights, credentials, or absolute workstation paths.
