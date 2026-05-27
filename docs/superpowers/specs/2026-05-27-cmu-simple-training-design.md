# CMU Simple Training Design

## Goal

Create a clean, independent training directory for CMU-MOSI and CMU-MOSEI. The
baseline extracts one feature vector per modality, maps each modality through a
linear head to a scalar logit, and averages the three logits for the final
prediction.

## Scope

The new code lives under `CMU_simple_v1/`. It does not reuse generated outputs,
caches, or data files from the existing `CMU_v1/` directory. Dataset files are
not committed or scaffolded into the new directory; command-line defaults leave
`--data_path` empty so the prepared local data can be supplied at runtime.

The implementation supports:

- `train_mosi.py` for CMU-MOSI.
- `train_mosei.py` for CMU-MOSEI.
- A shared dataset loader for aligned CMU pkl files.
- A shared three-modality classifier.
- Basic regression metrics for CMU sentiment evaluation.
- Lightweight smoke/unit tests that do not require the real datasets.

## Directory Layout

```text
CMU_simple_v1/
  README.md
  train_mosi.py
  train_mosei.py
  data/
    __init__.py
    cmu_dataset.py
  models/
    __init__.py
    classifier.py
    sequence_encoder.py
    text_encoder.py
  utils/
    __init__.py
    logger.py
    metrics.py
    seed.py
  tests/
    conftest.py
    test_classifier.py
    test_dataset.py
    test_metrics.py
```

## Data Contract

The dataset loader expects a pickle file containing split keys:

```python
{
    "train": [record, ...],
    "dev": [record, ...],
    "test": [record, ...],
}
```

Each record follows the existing CMU aligned format:

```python
((words, vision, audio), label, meta)
```

`words` is a token list or text-like sequence, `vision` and `audio` are
time-series feature arrays, `label` is a scalar or scalar-shaped array, and
`meta` is preserved for traceability. The loader pads or truncates vision and
audio to `max_seq_len`, tokenizes text through a HuggingFace tokenizer to
`bert_max_len`, and returns masks for all variable-length inputs.

## Model

The classifier has three independent branches:

```text
vision -> SequenceEncoder -> pooling -> Linear(hidden_sz, 1)      -> vision_logit
audio  -> SequenceEncoder -> pooling -> Linear(hidden_sz, 1)      -> audio_logit
text   -> BERT          -> pooling -> Linear(text_hidden_sz, 1)   -> text_logit
```

The final prediction is:

```python
final_logit = (vision_logit + audio_logit + text_logit) / 3.0
```

There is no concatenation fusion, cross-modal attention, gating, or nonlinear
projection between the pooled modality features and the linear heads.

## Text Encoder

Text uses `transformers.BertModel.from_pretrained(args.bert_model_name)`.
`--bert_model_name` can be either a model name such as `bert-base-uncased` or a
local pretrained-model path. `--freeze_bert` freezes all BERT parameters when
set. The default pooling strategy uses the BERT `[CLS]` vector for text.

## Vision And Audio Encoder

Vision and audio use a lightweight sequence encoder: a 1D convolutional
projection followed by a small Transformer encoder. Both branches are separate
instances with independent parameters. The encoder returns a sequence, then the
classifier pools it into one vector per modality.

Pooling supports:

- `last`: use the last time step for all modalities.
- `mean`: use mask-aware mean pooling for vision/audio and `[CLS]` for text.

The default is `mean` because padded CMU sequences make mask-aware pooling less
fragile than blindly taking the final padded position.

## Training

Both entry scripts share the same training behavior and only differ in dataset
defaults:

- MOSI defaults: `dataset=mosi`, `vision_dim=47`, `audio_dim=74`.
- MOSEI defaults: `dataset=mosei`, `vision_dim=35`, `audio_dim=74`.

Training uses regression loss. For each batch:

```python
loss = loss_fn(final_logit, label)
     + loss_fn(vision_logit, label)
     + loss_fn(audio_logit, label)
     + loss_fn(text_logit, label)
```

The default loss is L1 loss, with a CLI option to switch to MSE. Validation uses
only `final_logit`. The best checkpoint is selected by validation Acc2, and the
test split is evaluated after training with the best checkpoint.

## Outputs

Each run writes under:

```text
CMU_simple_v1/savepath/<dataset>/<run_name>/
```

The run directory contains:

- `training.log`
- `model_best.pt`
- `final_results.json`

The summary JSON includes best validation metrics and final test metrics.

## Metrics

Metrics are computed from scalar regression predictions:

- MAE
- Pearson correlation
- Acc7 after clipping and rounding to `[-3, 3]`
- Acc2 using non-zero binary sentiment polarity
- F1 using non-zero binary sentiment polarity

Degenerate correlation cases return `0.0` instead of propagating `NaN`.

## Error Handling

The dataset loader raises clear errors for missing `--data_path`, missing pkl
files, missing split keys, empty splits, and feature-dimension mismatches. BERT
load errors include guidance to pass a local model path when the environment has
no network access.

## Testing

Tests use synthetic data and small fake encoders where needed so they do not
require the real MOSI/MOSEI files or downloading BERT. Coverage focuses on:

- Dataset padding, truncation, masks, labels, and tokenizer integration.
- Metric behavior including degenerate correlation.
- Classifier forward output shapes and exact logit averaging.

## Non-Goals

This baseline does not implement robust-noise evaluation, cross-modal fusion
modules, feature caching, distributed training, mixed precision, or automatic
dataset download.
