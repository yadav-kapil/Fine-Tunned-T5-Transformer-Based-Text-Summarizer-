# Fine-Tuned T5 Transformer Dialogue Summarizer

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2%20Ubuntu-232F3E?style=flat-square&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/ec2/)
[![Nginx](https://img.shields.io/badge/Nginx-Reverse%20Proxy-009639?style=flat-square&logo=nginx&logoColor=white)](https://nginx.org/)
[![SSL](https://img.shields.io/badge/SSL-Let's%20Encrypt-003A70?style=flat-square&logo=letsencrypt&logoColor=white)](https://letsencrypt.org/)

An end-to-end Machine Learning project exploring **Transfer Learning and Domain-Specific Fine-Tuning** of a Seq2Seq Transformer for abstractive dialogue summarization. The project takes Google's pre-trained **T5-Small** architecture, fine-tunes it on 14,732 real-world conversational dialogues (SAMSum corpus), and deploys the resulting inference service to **AWS EC2** behind **Nginx**, **Docker**, and a **GitHub Actions CI/CD** pipeline — covering the complete journey from raw research notebook to a live, HTTPS-secured web application.

🌐 **Live Application:** [https://fine-tunned-t5transformer-summarizer.duckdns.org](https://fine-tunned-t5transformer-summarizer.duckdns.org)  
📖 **Interactive API Docs:** [https://fine-tunned-t5transformer-summarizer.duckdns.org/docs](https://fine-tunned-t5transformer-summarizer.duckdns.org/docs)

---

## Table of Contents

- [Fine-Tuning Overview & AI Architecture](#fine-tuning-overview--ai-architecture)
  - [Why Fine-Tune for Dialogue?](#why-fine-tune-for-dialogue)
  - [The Conversational NLP Challenge](#the-conversational-nlp-challenge)
  - [The SAMSum Dataset](#the-samsum-dataset)
  - [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
  - [Seq2Seq Tokenization & Label Formatting](#seq2seq-tokenization--label-formatting)
  - [Loss Function & Training Dynamics](#loss-function--training-dynamics)
  - [Inference Decoding Strategy](#inference-decoding-strategy)
- [Training Hyperparameters](#training-hyperparameters)
- [Before vs. After Fine-Tuning](#before-vs-after-fine-tuning)
- [System Architecture & Deployment Strategy](#system-architecture--deployment-strategy)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Local Setup & Quick Start](#local-setup--quick-start)
- [License](#license)

---

## Fine-Tuning Overview & AI Architecture

### Why Fine-Tune for Dialogue?

Pre-trained language models like T5 are trained on broad, general-purpose corpora (in T5's case, the 750 GB C4 dataset) using a **span-corruption objective** — masking random spans of text and training the model to reconstruct them. This builds rich linguistic representations but no task-specific understanding.

Dialogue summarization poses a fundamentally different challenge compared to news summarization (the standard benchmark domain). News articles follow the **inverted pyramid** structure: the most important information appears in the opening sentence, and the remaining text elaborates. A model can skim the top and produce a decent extractive summary.

Conversational dialogues have no such structure. Information is **distributed, implicit, and speaker-dependent**:

- There is no topic sentence. Meaning emerges across turns, not within them.
- Factual content is scattered through short, fragmented exchanges between multiple participants.
- Colloquial language, typos, interruptions, and implicit agreements ("sounds good", "done", "see you then") carry significant meaning that a formally-trained model tends to ignore or misinterpret.
- The model must resolve **pronouns across speakers** ("she said she'd pick it up" — which *she*? pick *what* up?) — a problem known as **coreference resolution** that standard summarization models are never trained to handle.

Fine-tuning on conversational data teaches the model to shift its behavior: rather than skim-and-extract, it must **read the full exchange, track participants, and synthesize a coherent third-person narrative** that captures the essential outcome of the conversation.

```text
Raw Dialogue:
  Amanda: I baked cookies. Do you want some?
  Jerry:  Sure! Chocolate chip please! Are you at home?
  Amanda: Yes, leaving for campus in 20 mins. I can drop them at your dorm.
  Jerry:  That would be awesome, thanks Amanda!

Generated Abstractive Summary:
  amanda baked cookies and will drop them off at jerry's dorm before leaving for campus.
```

The summary is not extracted from any single sentence — it is a **new, synthesized statement** that resolves speaker intent and compresses the exchange into a single coherent fact.

---

### The Conversational NLP Challenge

Abstractive summarization of multi-turn dialogue requires an **Encoder-Decoder** architecture — not a decoder-only causal model. The encoder must comprehend the full dialogue context before the decoder generates the summary.

```
Input Dialogue Tokens ──► [ T5 Encoder — Bidirectional Self-Attention ]
                                           │
                                   Cross-Attention Keys & Values
                                           ▼
Autoregressive Targets ──► [ T5 Decoder — Causal Self-Attention ] ──► Summary Tokens
```

**Encoder (Bidirectional Self-Attention):** Every token in the dialogue can attend to every other token simultaneously. This is critical for dialogue: a pronoun in turn 7 must be resolved against a noun in turn 2. Bidirectionality makes this possible — the encoder builds a rich, globally-aware contextual representation of the entire conversation.

**Decoder (Causal + Cross-Attention):** The decoder generates the summary autoregressively — one token at a time, left to right — conditioned on two sources of signal: the tokens it has already generated (causal self-attention), and the encoder's full dialogue representation (cross-attention). This cross-attention mechanism is the architectural bridge that lets the generated summary stay grounded in the source dialogue rather than drifting into hallucination.

---

### The SAMSum Dataset

The model was fine-tuned on the **SAMSum Corpus**, developed by Samsung R&D Institute Poland and released through HuggingFace Datasets. The corpus consists of linguist-authored, messenger-style dialogues paired with human-written third-person summaries — specifically constructed to reflect real informal communication patterns rather than synthetically generated text.

| Split | Samples |
| :--- | :--- |
| Training | 14,732 dialogues |
| Validation | 818 dialogues |
| Test | 819 dialogues |

Each dialogue involves 2–4 speakers and covers everyday domains: coordinating plans, sharing workplace updates, casual check-ins, and customer interactions. The human-written summaries use **third-person declarative prose**, which is the style the fine-tuned model learns to replicate.

---

### Data Preprocessing Pipeline

Before tokenization, raw dialogue text is passed through a lightweight normalization function. SAMSum dialogues can contain residual HTML tags (from web scraping), inconsistent line endings, and collapsed multi-word whitespace — all of which fragment the token sequence unnecessarily and reduce training signal quality.

```python
import re

def clean_data(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)   # Strip any residual HTML markup
    text = re.sub(r"\r\n", " ", text)    # Normalize Windows-style line endings
    text = re.sub(r"\s+", " ", text)     # Collapse repeated whitespace to single space
    return text.strip().lower()
```

Both the input dialogues and reference summaries are cleaned through the same function before entering the tokenization step, ensuring the model never sees inconsistent representations of the same linguistic content.

---

### Seq2Seq Tokenization & Label Formatting

The model uses T5's **SentencePiece Unigram tokenizer** (32,100-token vocabulary), loaded via HuggingFace's `AutoTokenizer`. For sequence-to-sequence training, both the source (dialogue) and target (summary) sequences must be tokenized separately, then assembled into the input-label format that the `Seq2SeqTrainer` expects.

```python
def tokenize(data):
    # Encode the source dialogue (encoder input)
    inputs = tokenizer(
        data["dialogue"],
        max_length=512,
        padding="max_length",
        truncation=True
    )

    # Encode the target summary (decoder supervision signal)
    targets = tokenizer(
        data["summary"],
        max_length=150,
        padding="max_length",
        truncation=True
    )

    # Labels are the token IDs the decoder is trained to predict
    inputs["labels"] = targets["input_ids"]
    return inputs
```

**Input cap (512 tokens):** Covers over 99% of SAMSum dialogues in full without truncation, preserving the complete conversational context the encoder needs.

**Output cap (150 tokens):** Provides sufficient budget for well-formed third-person summaries while imposing a hard upper bound that discourages the model from generating run-on or repetitive output — a known failure mode in seq2seq decoding.

---

### Loss Function & Training Dynamics

**Objective:** Standard **token-level Cross-Entropy Loss** computed at every position in the target sequence. The model is penalized proportionally for predicting low probability on the correct next token — no special seq2seq-specific loss is needed.

**Optimizer:** **AdamW** — the standard Adam optimizer with decoupled weight decay (`0.01`). Weight decay applies L2 regularization directly to the parameter weights (not the first/second moment estimates), which is the correction that makes it effective for Transformers.

**Learning Rate Schedule:** Starts at `3e-4` and ramps up linearly over the first **300 warmup steps**. This gradual warmup prevents large gradient updates in the early training phase when the randomly-initialized task head has not yet settled. After the warmup phase, the learning rate decays linearly back to zero over the remaining training steps.

**Checkpoint Selection:** The model is evaluated on the validation set at the end of each epoch. The checkpoint with the lowest validation loss is preserved and loaded for final inference — avoiding the common pitfall of deploying the last checkpoint rather than the best one.

> **Final Validation Loss: `0.315`** — indicating strong convergence and confident token predictions on held-out conversational examples.

---

### Inference Decoding Strategy

At inference time, selecting the single highest-probability token at each decoding step (**greedy search**) is fast but prone to two failure modes: getting trapped in repetitive loops, and committing too early to a locally-probable but globally-suboptimal phrasing.

This project uses **Beam Search** to explore a wider hypothesis space before committing to output tokens:

```python
summary_ids = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=150,
    num_beams=4,
    early_stopping=True
)
```

The decoder maintains **4 candidate sequences simultaneously** at each step. Rather than greedily selecting the best single next token, it tracks the 4 most promising partial hypotheses (ranked by cumulative log-probability) and continues expanding all of them in parallel. `early_stopping=True` terminates the search as soon as all 4 beams have produced an end-of-sequence token — avoiding unnecessary computation once all candidates are complete.

The result is a noticeably more fluent and coherent summary compared to greedy output, at the cost of 4× the autoregressive decoding steps (a worthwhile trade for a generation task where output quality matters).

---

## Training Hyperparameters

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Base Model** | `google-t5/t5-small` | 60.5M parameters — compact enough for single-GPU fine-tuning, capable enough for abstractive generation |
| **Dataset** | SAMSum Corpus | Purpose-built conversational summarization benchmark |
| **Training Samples** | 14,732 | Full training split, no downsampling |
| **Epochs** | `3` | Sufficient for convergence on this corpus size without overfitting |
| **Per-Device Batch Size** | `16` | Maximizes throughput on available unified memory |
| **Learning Rate** | `3e-4` | Recommended starting point for T5 fine-tuning; decays to zero |
| **Warmup Steps** | `300` | Linear ramp to stabilize early gradient updates |
| **Weight Decay** | `0.01` | Decoupled L2 regularization on non-bias parameters |
| **Evaluation Strategy** | Per epoch | Tracks validation loss after every complete pass |
| **Checkpoint Strategy** | Per epoch (`load_best_model_at_end=True`) | Ensures best-performing checkpoint is used for inference |
| **Max Input Length** | `512 tokens` | Fits 99%+ of SAMSum dialogues without truncation |
| **Max Output Length** | `150 tokens` | Caps summary length to prevent run-on generation |
| **Final Validation Loss** | **`0.315`** | Strong convergence on held-out dialogues |
| **Inference Beam Width** | `4` | Balanced quality / compute trade-off |
| **Training Accelerator** | Apple MPS | PyTorch Metal Performance Shaders (`torch.device("mps")`) |
| **Hardware** | MacBook Air M4 — 16 GB Unified Memory | Consumer hardware, no cloud GPU required |
| **Total Training Time** | **3 hours 17 minutes** | Full 3-epoch run |

---

## Before vs. After Fine-Tuning

The qualitative difference between the base pre-trained model and the fine-tuned version illustrates precisely why task-specific adaptation matters. The base model, pre-trained on span corruption, has no notion of what a "summary" of a conversation should look like. The fine-tuned model has internalized the dialogue-to-summary mapping from thousands of examples.

| Input Dialogue | Base T5-Small (No Fine-Tuning) | Fine-Tuned (This Project) |
| :--- | :--- | :--- |
| *Dan: Can you grab milk on your way back?*<br>*Kate: Sure, whole or skim?*<br>*Dan: Whole please! Thanks.* | **"milk"** — Outputs a single keyword fragment. No understanding of speaker intent or the request being made. | **"dan asks kate to buy whole milk on her way back."** — Correctly attributes the request to the right speaker and captures the action. |
| *Tom: Are you ready for the flight?*<br>*Mia: Almost, packing my passport now.*<br>*Tom: Uber arrives in 15 mins!* | **"flight"** — Identifies a topic keyword but produces no synthesized meaning. | **"mia is packing her passport. tom says the uber arrives in 15 minutes."** — Synthesizes two separate facts from two different speakers into a coherent, ordered summary. |

The base model treats dialogue as generic fill-in-the-blank text. The fine-tuned model understands that a conversation has **participants, a sequence of events, and an extractable outcome** — and it generates a summary that reflects all three.

---

## System Architecture & Deployment Strategy

```text
[ Web Browser / API Client ]
             │
             │  HTTPS (Port 443) 🔒  — TLS terminated by Let's Encrypt certificate
             ▼
[ Nginx Reverse Proxy  ·  AWS EC2  ·  Ubuntu 22.04 ]
             │
             │  proxy_pass http://127.0.0.1:7860  (internal, not publicly exposed)
             ▼
[ Docker Container: fine-tuned-t5-summarizer ]
             │
             ├── FastAPI Application Server  (app.py)
             ├── PyTorch 2.x — CPU-Optimized Wheel
             └── Fine-Tuned Weights  (models/model.safetensors  ·  231 MB via Git LFS)
```

### Deployment Design Decisions

**1. CPU-Only Docker Image**  
The inference target is a `t3.micro` EC2 instance with no GPU. Pulling the standard PyTorch wheel would add unused NVIDIA CUDA libraries and balloon the image to 4+ GB. The `Dockerfile` explicitly installs the CPU-only wheel from `https://download.pytorch.org/whl/cpu`, keeping the final image under 900 MB — a 75% reduction that significantly speeds up CI/CD build times and cold-start latency.

**2. `torch.inference_mode()` for Inference**  
All forward passes are wrapped in `torch.inference_mode()`. Unlike `torch.no_grad()`, inference mode additionally disables version tracking for `autograd`, resulting in lower memory overhead and faster token generation — particularly beneficial on memory-constrained instances.

**3. Self-Hosted GitHub Actions Runner on EC2**  
Rather than pushing a built image to a container registry and pulling it back down, the CI/CD pipeline runs directly on the EC2 instance as a self-hosted GitHub Actions runner. On every push to `main`:
- Pulls the updated source code and 231 MB model weights via Git LFS.
- Builds the Docker image locally on the instance.
- Stops the running container, starts the new one with `--restart unless-stopped`.
- Prunes Docker build cache and wipes the runner's workspace to maintain strict disk hygiene on the 10 GB EBS volume.

**4. Nginx + Let's Encrypt SSL**  
Port 7860 (the FastAPI/Docker port) is not publicly exposed. Nginx listens on Port 80 and 443, terminates the TLS handshake using an auto-renewing Let's Encrypt certificate, and proxies all traffic to `127.0.0.1:7860` over the internal loopback interface. This keeps the application container isolated from the public network while delivering standard HTTPS to clients.

---

## Project Structure

```text
Text-Summarizer-App/
├── .github/
│   └── workflows/
│       └── deploy.yml            # GitHub Actions CI/CD — self-hosted EC2 runner
├── datasets/
│   ├── samsum-train.csv          # Training split — 14,732 dialogue-summary pairs
│   ├── samsum-validation.csv     # Validation split — 818 pairs
│   └── samsum-test.csv           # Test split — 819 pairs
├── models/                       # Fine-tuned weights and tokenizer artifacts
│   ├── config.json               # T5 architecture configuration
│   ├── generation_config.json    # Beam search and generation constraints
│   ├── model.safetensors         # Trained model weights (231 MB, tracked by Git LFS)
│   ├── tokenizer.json            # HuggingFace fast tokenizer
│   └── tokenizer_config.json     # Vocabulary metadata and special token mappings
├── notebooks/
│   ├── text-summarizer.ipynb     # End-to-end fine-tuning pipeline
│   └── test-model.ipynb          # Local evaluation and inference validation
├── static/
│   ├── app.js                    # Async UI logic, keyboard shortcuts, and state management
│   └── style.css                 # Clean, responsive styling
├── templates/
│   └── index.html                # Semantic single-page application interface
├── .dockerignore                 # Excludes datasets and notebooks from the Docker build context
├── .gitattributes                # Git LFS tracking rules for large model files
├── .gitignore                    # Python and environment exclusion rules
├── Dockerfile                    # CPU-optimized production container definition
├── app.py                        # FastAPI application and inference engine
└── requirements.txt              # Pinned production dependencies
```

---

## API Reference

### Health Check

```
GET /health
```

Returns the current service status and the active compute device.

```json
{
  "status": "healthy",
  "device": "cpu"
}
```

### Summarize Conversation

```
POST /summarize
Content-Type: application/json
```

**Request Body:**
```json
{
  "dialogue": "Amanda: I baked cookies. Do you want some?\nJerry: Sure! Chocolate chip please! Are you at home?\nAmanda: Yes, leaving for campus in 20 mins. I can drop them at your dorm.\nJerry: That would be awesome, thanks Amanda!"
}
```

**Response:**
```json
{
  "summary": "amanda baked cookies and will drop them off at jerry's dorm before leaving for campus."
}
```

The `/summarize` endpoint accepts any multi-turn dialogue formatted as `Speaker: message` lines separated by newlines. The model handles 2–4 speakers and dialogues up to approximately 512 tokens (roughly 350–400 words of conversational text).

---

## Local Setup & Quick Start

### 1. Clone with Git LFS

The fine-tuned model weights (`model.safetensors`, 231 MB) are tracked by Git LFS. Run `git lfs pull` after cloning to download them:

```bash
git clone https://github.com/yadav-kapil/Fine-Tunned-T5-Transformer-Based-Text-Summarizer-.git
cd Fine-Tunned-T5-Transformer-Based-Text-Summarizer-
git lfs pull
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

The server starts on **[http://127.0.0.1:7860](http://127.0.0.1:7860)**. Open that URL in your browser to use the web interface, or hit the `/summarize` endpoint directly with any HTTP client.

> **Note:** First startup takes a few seconds as PyTorch loads and maps the model weights into memory. Subsequent requests respond in under 2 seconds on a modern CPU.

---

## License

Distributed under the **MIT License**. See `LICENSE` for details.
