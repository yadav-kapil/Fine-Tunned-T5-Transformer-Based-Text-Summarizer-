# Dialogue Summarizer

A learning project to understand **Transformer fine-tuning** end-to-end — from raw dataset preprocessing and sequence-to-sequence training, to saving a model and wrapping it in a FastAPI web application.

The model is a **T5-small** fine-tuned on the **SAMSum** conversational dataset to summarize multi-turn dialogues into concise third-person summaries.

---

## What I Learned

- How **Transformer models** (specifically encoder-decoder seq2seq) work for abstractive text summarization.
- How to **fine-tune a pre-trained T5** using HuggingFace `Trainer` on a custom dataset.
- **Tokenization** for seq2seq tasks — tokenizing both inputs and labels, padding, truncation, and setting the `labels` field.
- Running training on **Apple Silicon (MPS)** via PyTorch's Metal backend.
- How to **save and reload a fine-tuned model** (`save_pretrained`, `from_pretrained`).
- Serving a trained deep learning model through a **FastAPI REST endpoint**.
- Building a minimal **single-page web interface** to interact with the model.

---

## Model & Training

| | |
| :--- | :--- |
| **Base Model** | `google-t5/t5-small` (60M parameters) |
| **Task** | Abstractive dialogue summarization (seq2seq) |
| **Dataset** | SAMSum (14,732 train / 818 val / 819 test samples) |
| **Epochs** | 3 |
| **Batch Size** | 16 (train & eval) |
| **Learning Rate** | `3e-4` with 300 warmup steps |
| **Weight Decay** | `0.01` |
| **Input Length** | 512 tokens (dialogue) |
| **Output Length** | 150 tokens (summary) |
| **Final Val Loss** | 0.315 |
| **Hardware** | Apple MacBook Air M4, 16GB Unified Memory |
| **Accelerator** | Apple Silicon MPS (`torch.device("mps")`) |
| **Training Time** | **3 hours 17 minutes** |

**Generation** at inference uses beam search (`num_beams=4`, `early_stopping=True`).

---

## Project Structure

```text
Text-Summarizer-App/
├── datasets/
│   ├── samsum-train.csv          # 14,732 dialogue-summary pairs
│   ├── samsum-validation.csv     # 818 samples
│   └── samsum-test.csv           # 819 samples
├── models/                       # Saved fine-tuned model
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors         # Trained weights (231 MB)
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── notebooks/
│   ├── text-summarizer.ipynb     # Full training pipeline
│   └── test-model.ipynb          # Inference & testing
├── static/
│   ├── app.js                    # Async form submission & UI logic
│   └── style.css
├── templates/
│   └── index.html                # Side-by-side web interface
├── app.py                        # FastAPI server
├── requirements.txt
├── .gitattributes                # Git LFS for model weights
└── .gitignore
```

---

## Training Pipeline (`text-summarizer.ipynb`)

**1. Load & Inspect Data**
```python
import pandas as pd

train_data = pd.read_csv("../datasets/samsum-train.csv")
val_data = pd.read_csv("../datasets/samsum-validation.csv")
```

**2. Clean & Preprocess**
```python
import re

def clean_data(text):
    text = re.sub(r"<.*?>", " ", text)   # strip HTML tags
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

train_data["dialogue"] = train_data["dialogue"].apply(clean_data)
train_data["summary"]  = train_data["summary"].apply(clean_data)
```

**3. Tokenize for Seq2Seq**
```python
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize(data):
    inputs  = tokenizer(data["dialogue"], max_length=512, padding="max_length", truncation=True)
    targets = tokenizer(data["summary"],  max_length=150, padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_data = train_data.apply(tokenize, axis=1).tolist()
val_data   = val_data.apply(tokenize, axis=1).tolist()
```

**4. Load Model & Train**
```python
from transformers import T5ForConditionalGeneration, TrainingArguments, Trainer

model = T5ForConditionalGeneration.from_pretrained("t5-small", device_map="auto")

training_args = TrainingArguments(
    output_dir="../model_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=300,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    dataloader_pin_memory=False      # required for MPS on Mac
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data
)

trainer.train()
```

**5. Save**
```python
trainer.save_model("../models")
tokenizer.save_pretrained("../models")
```

---

## Inference (`test-model.ipynb`)

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("../models")
model = AutoModelForSeq2SeqLM.from_pretrained("../models").to(device)

def summarize_dialogue(dialogue):
    inputs = tokenizer(dialogue, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
    targets = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(targets[0], skip_special_tokens=True)
```

**Example:**
```
Input:
  Amanda: I baked cookies. Do you want some?
  Jerry: Sure! Chocolate chip please! Are you at home?
  Amanda: Yes, leaving for campus in 20 mins. I can drop them at your dorm.
  Jerry: That would be awesome, thanks Amanda!

Output:
  amanda baked cookies and will drop them off at jerry's dorm before leaving for campus.
```

---

## Running the Web App

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** — paste any conversation and click **Summarize**.

The `/summarize` POST endpoint accepts both JSON and HTML Form payloads and returns `{"summary": "..."}`.
