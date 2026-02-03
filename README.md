project_purpose_md = """
This project demonstrates Named Entity Recognition (NER) using the CoNLL-2003 dataset and a fine-tuned BERT-based model. NER is a crucial task in natural language processing that identifies and classifies named entities in text into predefined categories such as person names, organizations, locations, and more.
"""

required_libraries_md = """
### Required Libraries

The following libraries are required to run this project:
- `datasets`: For loading and managing the CoNLL-2003 dataset.
- `seqeval`: For evaluating the performance of the NER model.
- `transformers`: For using pre-trained models, tokenizers, and the `Trainer` API from Hugging Face.
- `torch`: The underlying deep learning framework.
- `numpy`: For numerical operations.

Installation:
```bash
!pip install datasets --no-build-isolation
!pip install seqeval
!pip install transformers[torch]
```
"""

dataset_description_md = """
### CoNLL-2003 Dataset Description

The CoNLL-2003 dataset is widely used for training and evaluating Named Entity Recognition (NER) models. The dataset focuses on four types of named entities: persons (PER), locations (LOC), organizations (ORG), and miscellaneous entities (MISC).

#### Dataset Structure:
Each data file contains four columns separated by a single space:
1. Word
2. Part-of-Speech (POS) tag
3. Syntactic chunk tag
4. Named entity tag

Words are listed on separate lines, and sentences are separated by a blank line.
The chunk and named entity tags follow the IOB2 tagging scheme:
- `B-TYPE`: Beginning of a phrase of type TYPE
- `I-TYPE`: Inside a phrase of type TYPE
- `O`: Outside any named entity phrase

#### Example:
```python
{
    "chunk_tags": [11, 12, 12, 21, 13, 11, 11, 21, 13, 11, 12, 13, 11, 21, 22, 11, 12, 17, 11, 21, 17, 11, 12, 12, 21, 22, 22, 13, 11, 0],
    "id": "0",
    "ner_tags": [0, 3, 4, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "pos_tags": [12, 22, 22, 38, 15, 22, 28, 38, 15, 16, 21, 35, 24, 35, 37, 16, 21, 15, 24, 41, 15, 16, 21, 21, 20, 37, 40, 35, 21, 7],
    "tokens": ["The", "European", "Commission", "said", "on", "Thursday", "it", "disagreed", "with", "German", "advice", "to", "consumers", "to", "shun", "British", "lamb", "until", "scientists", "determine", "whether", "mad", "cow", "disease", "can", "be", "transmitted", "to", "sheep", "."]
}
```

#### Named Entity Tags
- **O**: Outside a named entity
- **B-PER**: Beginning of a person's name
- **I-PER**: Inside a person's name
- **B-ORG**: Beginning of an organization name
- **I-ORG**: Inside an organization name
- **B-LOC**: Beginning of a location name
- **I-LOC**: Inside a location name
- **B-MISC**: Beginning of miscellaneous entity
- **I-MISC**: Inside a miscellaneous entity
"""

model_architecture_md = """
### Model Architecture

A `bert-base-cased` model from the Hugging Face Transformers library is used for token classification. This model is a pre-trained BERT variant that has been cased (distinguishes between \"hello\" and \"Hello\") and is suitable for various downstream NLP tasks including Named Entity Recognition. The final layer of the model is adapted for token classification with `num_labels=9`, corresponding to the number of unique NER tags in the CoNLL-2003 dataset (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC).

The model is initialized with pre-trained weights and fine-tuned on the CoNLL-2003 dataset.
"""

tokenization_labeling_md = """
### Tokenization and Label Alignment

The `AutoTokenizer` from `transformers` is used with the `bert-base-cased` checkpoint to convert text into token IDs. A custom function `tokenize_and_align_labels` is implemented to handle subword tokenization and align labels correctly.

#### Token Labeling in NER: Use of `-100`

In Named Entity Recognition (NER) tasks, the label `-100` is commonly used to signify that certain tokens should be ignored during the loss calculation in model training. This approach helps focus the learning on meaningful parts of the data. Here's an overview of the types of tokens typically assigned a `-100` label:

### 1. **Subsequent Sub-tokens**
After a word is split into multiple sub-tokens, only the first sub-token receives the actual entity label. Subsequent sub-tokens receive `-100` to ensure that entity labels are not incorrectly assigned to fragments of words.

### 2. **Special Tokens**
Special tokens such as `[CLS]`, `[SEP]`, and `[PAD]` used for managing sequence boundaries and lengths in models like BERT are also assigned `-100` as they do not correspond to real words in the text.

### 3. **Non-Entity Tokens**
In certain training setups, tokens that do not correspond to any entity and are not the focus of the task might also be marked with `-100`, especially in cases of imbalanced datasets.

#### Example
- **Sentence**: \"John lives in New York\"
- **Tokens**: `[\"[CLS]\", \"John\", \"lives\", \"in\", \"New\", \"York\", \"[SEP]\"]`
- **Labels**: `[-100, \"B-PER\", \"O\", \"O\", \"B-LOC\", \"I-LOC\", -100]`

This labeling strategy is critical for efficient model training, ensuring that the model focuses only on relevant tokens. The `DataCollatorForTokenClassification` is used to dynamically pad inputs to the longest sequence in a batch.
"""

training_process_md = """
### Training Process

The model is fine-tuned using the Hugging Face `Trainer` API. Key training arguments include:
- **`output_dir`**: `./results` (directory to save model checkpoints and logs)
- **`eval_strategy`**: `epoch` (evaluation performed at the end of each epoch)
- **`learning_rate`**: `2e-5`
- **`per_device_train_batch_size`**: `16`
- **`per_device_eval_batch_size`**: `16`
- **`num_train_epochs`**: `3`
- **`weight_decay`**: `0.01`

The `Trainer` is initialized with the fine-tuned model, training arguments, tokenized datasets (train and validation), tokenizer, data collator, and the custom `compute_metrics` function.
"""

evaluation_process_md = """
### Evaluation Process

The `seqeval` library is used to compute standard metrics for sequence labeling tasks (precision, recall, F1-score, and accuracy).

#### `compute_metrics` Function:
The `compute_metrics` function processes model predictions and true labels:
1. It converts raw predictions (logits) to predicted label IDs using `argmax`.
2. It filters out special tokens and ignored labels (`-100`) from both true and predicted labels.
3. It maps label IDs back to their string representations (e.g., 0 to \"O\", 3 to \"B-PER\").
4. It then uses `metric.compute()` from `seqeval` to calculate overall precision, recall, F1-score, and accuracy.
"""

evaluation_results_md = """
### Evaluation Results

After 3 epochs of training, the model achieves the following performance on the validation set:

```json
{
    \"eval_loss\": 0.03746436536312103,
    \"eval_precision\": 0.9418546365914787,
    \"eval_recall\": 0.9486704813194211,
    \"eval_f1\": 0.9452502724909868,
    \"eval_accuracy\": 0.9909076749347767,
    \"eval_runtime\": 10.5735,
    \"eval_samples_per_second\": 307.372,
    \"eval_steps_per_second\": 19.294,
    \"epoch\": 3.0
}
```

The model demonstrates strong performance with an F1-score of approximately `0.945` and an accuracy of `0.991`, indicating effective Named Entity Recognition capabilities on the CoNLL-2003 dataset.
"""

readme_content = """
# Named Entity Recognition Project

## Introduction

""" + project_purpose_md + """

## Setup

""" + required_libraries_md + """

## Dataset

""" + dataset_description_md + """

## Model

""" + model_architecture_md + """

## Tokenization and Labeling

""" + tokenization_labeling_md + """

## Training

""" + training_process_md + """

## Evaluation

""" + evaluation_process_md + """

## Results

""" + evaluation_results_md

print("README content successfully assembled into `readme_content` variable.")
