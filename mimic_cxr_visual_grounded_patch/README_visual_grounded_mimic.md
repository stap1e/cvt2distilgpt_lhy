# MIMIC-CXR CvT2DistilGPT2 visual-grounding first version

## Files

- `cvt2distilgpt2_mimic_cxr_visual_grounded.py`: new model module.
- `train_mimic_cxr_visual_grounded.yaml`: matching training configuration.
- `stages_visual_grounded.py`: permits non-strict warm-starting from a baseline checkpoint.

Place the Python module beside the original
`cvt2distilgpt2_mimic_cxr_chen.py`, place `stages_visual_grounded.py` beside
the existing stages module, and place the YAML under `config/`.

## What the code changes

For the same teacher-forced report prefix, it evaluates the target clinical
words under:

1. the correct image;
2. a zero visual memory;
3. a report-dissimilar image selected from the current mini-batch.

It enforces the correct-image target log probability to exceed each negative
by `visual_margin`. It also adds a pooled-image auxiliary head that predicts
positive, negative, or uncertain states weakly extracted from the report.

This is designed to reduce the decoder's ability to minimise CE while ignoring
visual memory. It does **not** yet claim anatomy-level grounding because the
provided files do not include region boxes or claim-to-region annotations.

## Important memory change

The original model needs one encoder/decoder forward per CE step. This version
usually needs three decoder forwards (correct, null, mismatch), so the supplied
YAML lowers `mbatch_size` from 16 to 4. Reduce it further if CUDA memory is
insufficient, or use gradient accumulation in the trainer configuration.

## Recommended ablations

1. Baseline CE.
2. CE + null margin.
3. CE + mismatch margin.
4. CE + both margins.
5. CE + both margins + concept head.
6. Full model + nonclinical invariance.

Besides CIDEr and CheXbert, compare `val_null_clinical_gap` and
`val_mismatch_clinical_gap`. A larger positive gap means ground-truth clinical
phrases become more dependent on the correct image.

## Known limitations

- The clinical parser is a built-in radiology lexicon, not RadGraph.
- In-batch mismatches may still share some findings.
- There is no anatomical region intervention yet.
- `no_repeat_ngram_size` prevents exact n-gram loops but not clinical synonyms.
- This first version intentionally supports `train_mode: ce` only. DPO or SCST
  should be applied after a visually grounded CE checkpoint is obtained.

## Warm-start from the baseline

Use the new stages module and provide the baseline checkpoint through the same
`warm_start_ckpt_path` mechanism already supported by the project. Non-strict
loading is necessary because the new concept head has no weights in the old
checkpoint.
