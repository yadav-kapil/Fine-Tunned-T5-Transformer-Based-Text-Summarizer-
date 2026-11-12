import os
import re
import sys
from pathlib import Path

try:
    import torchvision  # noqa: F401
except Exception:
    sys.modules["torchvision"] = None

import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import T5ForConditionalGeneration, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Fine-Tuned T5 Transformer Summarizer API",
    description="Production API serving a fine-tuned T5-small seq2seq model for dialogue summarization"
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained(str(MODELS_DIR)).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR))


def clean_data(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)
    if not dialogue:
        return ""

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(targets[0], skip_special_tokens=True)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(device)}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/summarize")
async def summarize(request: Request):
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = await request.json()
            dialogue = data.get("dialogue", "")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
    else:
        form = await request.form()
        dialogue = form.get("dialogue", "")

    if not dialogue or not dialogue.strip():
        return {"summary": "Please provide dialogue text to summarize."}

    return {"summary": summarize_dialogue(dialogue)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
