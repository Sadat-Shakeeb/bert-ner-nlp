# Named Entity Recognition (NER) with BERT (CoNLL-2003)

A simple, reproducible project demonstrating Named Entity Recognition (NER) using the CoNLL-2003 dataset and a fine-tuned BERT-based token classification model (bert-base-cased). The project uses Hugging Face Transformers, Datasets, and PyTorch, and evaluates using seqeval.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Dataset (CoNLL-2003)](#dataset-conll-2003)
- [Model & Tokenization](#model--tokenization)
- [Token Labeling & -100 Explanation](#token-labeling---100-explanation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results (example)](#results-example)

---

## Project Overview

This repository shows how to fine-tune a pre-trained BERT model for token-level classification on the CoNLL-2003 NER dataset. It includes tokenization with alignment of labels to sub-tokens, training using the `Trainer` API, and evaluation with `seqeval`.

---

## Features

- Uses `bert-base-cased` for token classification
- Handles subword tokenization and label alignment
- Training with Hugging Face `Trainer` and `DataCollatorForTokenClassification`
- Evaluation using `seqeval` (precision / recall / F1 / accuracy)
- Example evaluation metrics included

---

## Requirements

Install the required Python libraries:

```bash
pip install datasets --no-build-isolation
pip install seqeval
pip install transformers[torch]
pip install torch
pip install numpy
```

(You may want to create and use a virtual environment.)

---

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Sadat-Shakeeb/bert-ner-nlp.git
   cd bert-ner-nlp
   ```

2. Install dependencies (see Requirements section).

3. Prepare / download the CoNLL-2003 dataset (the `datasets` library makes this easy).

4. Run training script (example placeholder — replace with actual script name if different):
   ```bash
   python train.py --model_name_or_path bert-base-cased --dataset conll2003 --output_dir ./results
   ```

5. Evaluate / predict:
   ```bash
   python evaluate.py --model_dir ./results --dataset conll2003
   ```

(Adjust script names and CLI args to match the repo's actual scripts.)

---

## Dataset (CoNLL-2003)

CoNLL-2003 is a standard NER benchmark covering four entity types:
- PER — Person
- LOC — Location
- ORG — Organization
- MISC — Miscellaneous

Format (simplified):
- Each line: WORD POS CHUNK NER
- Sentences separated by blank lines
- The dataset uses the IOB2 scheme (B-*, I-*, O)

Example dataset entry (JSON-like excerpt):
```json
{
  "tokens": ["The", "European", "Commission", "said", ...],
  "pos_tags": [...],
  "chunk_tags": [...],
  "ner_tags": [...]
}
```

---

## Model & Tokenization

- Model: `bert-base-cased` (Hugging Face Transformers) for `AutoModelForTokenClassification`.
- Tokenizer: `AutoTokenizer` for the same checkpoint.
- The tokenizer may split words into sub-tokens (WordPiece). You must align the original word-level labels to token-level labels.

A typical pipeline:
1. Load dataset with `datasets`.
2. Create a `tokenize_and_align_labels` function which:
   - Tokenizes each sentence with `is_split_into_words=True`.
   - Maps word-level labels to token-level labels:
     - First token of a word → original label (e.g., `B-PER`).
     - Subsequent sub-tokens → `-100` to ignore in loss.
   - Special tokens (`[CLS]`, `[SEP]`, padding) → `-100`.

3. Use `DataCollatorForTokenClassification` to dynamically pad batches.

---

## Token Labeling & -100 Explanation

When a tokenizer splits a word into multiple sub-tokens, token classification models must not punish the model for predicting the same label on those sub-tokens. Common approach:

- Assign label for the first sub-token.
- Assign -100 to subsequent sub-tokens (these are ignored by PyTorch loss functions).
- Assign -100 to special tokens (`[CLS]`, `[SEP]`, `[PAD]`) because they don't correspond to actual words.

Example:
- Sentence: "John lives in New York"
- Tokens: `[CLS] John liv ##es in New York [SEP]`
- Labels: `[-100, B-PER, O, -100, O, B-LOC, I-LOC, -100]`

This ensures the loss is computed only for meaningful token positions.

---

## Training (example settings)

Example Hugging Face `TrainingArguments` used for fine-tuning:

- output_dir: `./results`
- eval_strategy: `epoch`
- learning_rate: `2e-5`
- per_device_train_batch_size: `16`
- per_device_eval_batch_size: `16`
- num_train_epochs: `3`
- weight_decay: `0.01`

Trainer setup usually includes:
- model (AutoModelForTokenClassification)
- training args
- train and eval datasets (tokenized)
- tokenizer
- data collator
- compute_metrics function (see Evaluation section)

---

## Evaluation

Use `seqeval` to compute precision, recall, F1, and accuracy for the sequence labeling task.

The `compute_metrics` function typically:
1. Converts model logits to predicted label IDs with `argmax`.
2. Iterates through the batch and filters out label positions with `-100` (these are ignored).
3. Maps label IDs back to label strings (e.g., `0 -> "O"`, etc.).
4. Calls `metric.compute()` from `seqeval` to calculate metrics.

---

## Results (example)

After training for 3 epochs (example run), evaluation metrics on validation:

```json
{
  "eval_loss": 0.03746436536312103,
  "eval_precision": 0.9418546365914787,
  "eval_recall": 0.9486704813194211,
  "eval_f1": 0.9452502724909868,
  "eval_accuracy": 0.9909076749347767,
  "eval_runtime": 10.5735,
  "eval_samples_per_second": 307.372,
  "eval_steps_per_second": 19.294,
  "epoch": 3.0
}
```

This example shows a high F1 (~0.945) and accuracy (~0.991) on CoNLL-2003 validation data.

---

