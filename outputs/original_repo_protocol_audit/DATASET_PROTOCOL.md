# Dataset Input Protocol

## Scope and sources

Read-only audit of dataset loading and preprocessing logic in:

- `cvt2distilgpt2_mimic_cxr_chen.py`
- `cvt2distilgpt2_iu_x_ray_chen.py`
- `tools/dataset/dataset.py`
- `tools/dataset/mimc_cxr_chen.py`
- `tools/dataset/iu_x_ray_chen.py`
- `tools/dataset/mimic_cxr_chen_tokenizer.py`
- `tools/dataset/iu_x_ray_chen_tokenizer.py`
- `tools/multi_image.py`

## 1. Supported datasets

The code path supports two Chen/R2Gen-style datasets:

1. **MIMIC-CXR Chen**
   - Model class: `CvT2DistilGPT2MIMICXRChen`
   - Dataset subset class: `tools.dataset.mimc_cxr_chen.TaskSubset`
   - Tokenizer/cleaner: `tools.dataset.mimic_cxr_chen_tokenizer.TokenizerChen`

2. **IU X-Ray Chen**
   - Model class: `CvT2DistilGPT2IUXRayChen`
   - Dataset subset class: `tools.dataset.iu_x_ray_chen.TaskSubset`
   - Tokenizer/cleaner: `tools.dataset.iu_x_ray_chen_tokenizer.TokenizerChen`

No other active dataset classes were found under `tools/dataset/` or `dataset/`. The root `dataset/` directory only contains `.gitkeep` placeholders for `iu_x-ray_chen`, `mimic_cxr_chen`, and `mimic_cxr_jpg`.

## 2. Dataset files and classes

| Dataset | Model file/class | Dataset class | Tokenizer class | Annotation path expected by source | Image root expected by source |
|---|---|---|---|---|---|
| MIMIC-CXR Chen | `cvt2distilgpt2_mimic_cxr_chen.py::CvT2DistilGPT2MIMICXRChen` | `tools/dataset/mimc_cxr_chen.py::TaskSubset` | `tools/dataset/mimic_cxr_chen_tokenizer.py::TokenizerChen` | `<dataset_dir>/mimic_cxr_chen/annotation.json` | `<dataset_dir>/mimic_cxr_chen/mimic_cxr_jpg/files` |
| IU X-Ray Chen | `cvt2distilgpt2_iu_x_ray_chen.py::CvT2DistilGPT2IUXRayChen` | `tools/dataset/iu_x_ray_chen.py::TaskSubset` | `tools/dataset/iu_x_ray_chen_tokenizer.py::TokenizerChen` | `<dataset_dir>/iu_x-ray_chen/annotation.json` | `<dataset_dir>/iu_x-ray_chen/images` |

Note: README uses `annotations.json`, but the source uses singular `annotation.json`.

## 3. Base Dataset `__init__` protocol

Both task-specific `TaskSubset` classes inherit `tools.dataset.dataset.Subset` and do not define their own `__init__`.

`Subset.__init__` parameters:

```python
def __init__(
    self,
    examples=None,
    transforms=None,
    colour_space=None,
    tokenizer=None,
    decoder_max_len=None,
    self_critical=False,
    train=False,
    add_bos_eos_manually=False,
    num_samples=None,
    sample_seed=43,
    **kwargs,
):
```

Meaning:

- `examples`: list of dicts after `format_examples()`.
- `transforms`: torchvision transform pipeline.
- `colour_space`: image conversion mode, used as `PIL.Image.convert(colour_space)`. The model passes `RGB`.
- `tokenizer`: GPT2 tokenizer used for decoder training inputs.
- `decoder_max_len`: max sequence length for decoder training/generation.
- `self_critical`: present but not used in current training flow except to skip tokenization if true.
- `train`: if true, `__getitem__` adds tokenized decoder inputs and labels.
- `add_bos_eos_manually`: if true, prepends `bos_token` and appends `eos_token` before GPT2 tokenization.
- `num_samples` / `sample_seed`: optional random subset sampling.
- `kwargs['normalisation']`: optional extra normalization hook; not used by these model constructors.

## 4. Annotation JSON expected format

The model code expects the annotation file to be JSON with top-level split keys:

```json
{
  "train": [],
  "val": [],
  "test": []
}
```

Each example must contain at least:

- `id`: sample identifier. This is later used as `batch['id']`, passed as `dicom_id` to `ReportLogger`, and as `ids` to metrics.
- `image_path`: list of relative image paths. The list semantics differ by dataset.
- `report`: raw target report string.

Fields **not required by the current source**:

- `findings`: not read.
- `impression`: not read.
- `subject_id`: not read.
- `study_id`: not read.
- `dicom_id`: not read as a separate field; `id` plays the role of identifier/dicom_id for logging.
- view labels such as frontal/lateral: not read.

If these fields exist in the annotation JSON, they may remain in each example dict unless popped, but the current `__getitem__` and metric code do not consume them.

## 5. Minimal annotation examples

### MIMIC-CXR Chen minimal example

```json
{
  "train": [
    {
      "id": "mimic_sample_0001",
      "image_path": ["p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"],
      "report": "The lungs are clear. No pleural effusion or pneumothorax."
    }
  ],
  "val": [
    {
      "id": "mimic_sample_0002",
      "image_path": ["p10/p10000032/s53189527/example.jpg"],
      "report": "Mild cardiomegaly. No focal consolidation."
    }
  ],
  "test": [
    {
      "id": "mimic_sample_0003",
      "image_path": ["p10/p10000032/s53911762/example.jpg"],
      "report": "No acute cardiopulmonary abnormality."
    }
  ]
}
```

The relative path is joined to `<dataset_dir>/mimic_cxr_chen/mimic_cxr_jpg/files`.

### IU X-Ray Chen minimal example

```json
{
  "train": [
    {
      "id": "iu_sample_0001",
      "image_path": ["CXR1_1_IM-0001-1001.png", "CXR1_1_IM-0001-2001.png"],
      "report": "The lungs are clear. The heart size is normal."
    }
  ],
  "val": [],
  "test": []
}
```

The relative paths are joined to `<dataset_dir>/iu_x-ray_chen/images`. IU `TaskSubset` indexes `image_file_path[0]` and `image_file_path[1]`, so each example must provide at least two paths.

## 6. Image path parsing

### Common `format_examples()` transformation

Both model classes perform the same field transformation:

```python
i["image_file_path"] = i.pop("image_path")
i["label"] = i.pop("report")
i["image_file_path"] = [os.path.join(self.dataset_dir, j) for j in i["image_file_path"]]
i["label"] = self.chen_tokenizer(i["label"])[:self.chen_max_seq_length]
i["label"] = self.chen_tokenizer.decode(i["label"][1:])
```

After this transformation, each example uses:

- `image_file_path`: list of absolute or joined image paths.
- `label`: cleaned/truncated report text.
- `id`: unchanged.

### MIMIC-CXR

- Source image root: `<dataset_dir>/mimic_cxr_chen/mimic_cxr_jpg/files`
- The annotation's `image_path` must be a list.
- `TaskSubset.__getitem__` loads only `example['image_file_path'][0]`.
- Additional image paths in the list are ignored by the current MIMIC dataset class.

### IU X-Ray

- Source image root: `<dataset_dir>/iu_x-ray_chen/images`
- The annotation's `image_path` must be a list with at least two entries.
- `TaskSubset.__getitem__` loads index `0` and index `1` and stacks them with `torch.stack((image_1, image_2), 0)`.

## 7. Single-image vs multi-image behavior

| Dataset | Annotation `image_path` | Loaded images per sample | `encoder_images` shape before batching | Batch shape with default collate |
|---|---|---|---|---|
| MIMIC-CXR | list of one or more relative paths | only first image | image tensor `[C,H,W]` | `[B,C,H,W]` |
| IU X-Ray | list of at least two relative paths | first two images | stacked tensor `[2,C,H,W]` | `[B,2,C,H,W]` |

`tools/multi_image.py` expects IU batches to have a views dimension at axis 1. It flattens `[B,V,C,H,W]` to `[B*V,C,H,W]`, then restores encoder outputs by concatenating features along the spatial axis.

## 8. View filtering

No explicit view filtering was found.

- There is no filtering by `frontal`, `lateral`, `AP`, `PA`, etc.
- MIMIC uses the first path in `image_path` regardless of view.
- IU uses the first two paths in list order.

Therefore, the annotation file is responsible for ordering/selecting images.

## 9. `__getitem__` return values

### MIMIC-CXR `TaskSubset.__getitem__`

Returns:

```python
{
  "id": example["id"],
  "encoder_images": image,
  "labels": example["label"],
  "image_filepaths": example["image_file_path"][0],
  # plus decoder fields during training
}
```

Training-only additions when `train=True` and `self_critical=False`:

```python
{
  "decoder_input_ids": Tensor,
  "decoder_attention_mask": Tensor,
  "label_ids": Tensor,
  # optional "decoder_token_type_ids" if tokenizer returns it
}
```

### IU X-Ray `TaskSubset.__getitem__`

Returns:

```python
{
  "id": example["id"],
  "encoder_images": torch.stack((image_1, image_2), 0),
  "labels": example["label"],
  # plus decoder fields during training
}
```

IU does **not** return `image_filepaths` in the current source.

## 10. Collate / batch protocol

No custom `collate_fn` is used in the model dataloaders. PyTorch default collation applies.

Dataloader construction:

```python
DataLoader(
    self.train_set or self.val_set or self.test_set,
    batch_size=self.mbatch_size,
    num_workers=self.num_workers,
    shuffle=True/False,
    prefetch_factor=self.prefetch_factor,
)
```

Expected batch keys:

Validation/test:

- `id`: default-collated list/sequence of IDs.
- `encoder_images`: tensor.
  - MIMIC: `[B,C,H,W]`
  - IU: `[B,2,C,H,W]`
- `labels`: list/sequence of cleaned report strings.
- MIMIC only: `image_filepaths`, if using MIMIC dataset class.

Training additionally:

- `decoder_input_ids`: tensor `[B, decoder_max_len]`
- `decoder_attention_mask`: tensor `[B, decoder_max_len]`
- `label_ids`: tensor `[B, decoder_max_len]`
- optional `decoder_token_type_ids`

## 11. Report cleaning and target text

The model does not train/evaluate against the raw `report` string directly. It cleans and truncates the report through the Chen tokenizer, then decodes it back into text.

Pipeline:

```text
raw annotation report
  -> TokenizerChen.clean_report_*()
  -> token IDs with thresholded vocabulary and <unk>
  -> truncate to chen_max_seq_length = 60
  -> drop leading 0 boundary token with [1:]
  -> decode IDs back to cleaned text
  -> example['label']
```

### MIMIC cleaner

`TokenizerChen.clean_report_mimic_cxr()`:

- replaces newlines with spaces.
- normalizes repeated underscores and repeated spaces.
- normalizes repeated periods.
- strips numbered prefixes like `1.`, `. 2.`, etc.
- lowercases.
- removes punctuation matched by regex `[.,?;*!%^&_+():-\[\]{}]` and removes quotes, slashes, backslashes, apostrophes.
- joins non-empty cleaned sentences with `" . "` and appends final `" ."`.

### IU cleaner

`TokenizerChen.clean_report_iu_xray()`:

- similar sentence/numbering cleanup to MIMIC.
- does not include MIMIC-specific newline/underscore normalization.
- lowercases and removes punctuation.
- joins sentences with `" . "` and appends final `" ."`.

## 12. Tokenizer construction and use

There are two tokenizer concepts:

1. **Chen tokenizer (`TokenizerChen`)**
   - Dataset-specific cleaner and vocabulary.
   - Vocabulary is built from the **training split only**.
   - Keeps tokens whose count is `>= threshold` and adds `<unk>`.
   - Assigns token IDs starting at 1; ID 0 is used as a boundary token in `__call__`.
   - Used only to normalize/truncate the target report text into `label`.

2. **GPT2 tokenizer (`transformers.GPT2TokenizerFast`)**
   - Loaded from `<ckpt_zoo_dir>/distilgpt2` with `local_files_only=True`.
   - Adds special tokens `{bos_token: "[BOS]", pad_token: "[PAD]"}`.
   - Used for decoder training inputs and decoding generated output.

Training target:

- Full cleaned `report` text only.
- No separate findings/impression fields are used.
- No concatenation logic exists for findings/impression in current source.

## 13. Dataset fields MedGemma-RRG should preserve for compatibility

To reuse original dataset/metrics/logging behavior, a new repository should support at least:

Required original-compatible fields:

- `id`: stable sample identifier. Original code treats this as the report logger `dicom_id` and metrics ID.
- `image_path`: list of relative image paths.
- `report`: raw full target report string.
- top-level splits: `train`, `val`, `test`.

Strongly recommended MedGemma additions for future work:

- `sample_id`: explicit stable sample ID, can equal original `id`.
- `dicom_id`: explicit DICOM/image identifier if available; original `id` can map here for legacy CSV.
- `study_id`: explicit study identifier if available.
- `subject_id`: explicit subject identifier if available.
- `image_paths`: normalized list of image paths in output files; can be derived from original `image_path`.
- `findings` / `impression`: optional raw sections if source data provides them, but mark optional because original code ignores them.
- `split`: useful in output records, not required by original annotations because split is encoded by top-level keys.

## 14. Compatibility risks

- The source expects `annotation.json`, not README's `annotations.json`.
- MIMIC silently ignores all images except the first path.
- IU assumes at least two paths and uses list order as the view protocol.
- `id` is overloaded: it is dataset sample ID, metric ID, and report CSV `dicom_id`.
- Original report outputs do not include reference text or image paths, so a MedGemma output protocol should add them while optionally also writing the legacy CSV.
