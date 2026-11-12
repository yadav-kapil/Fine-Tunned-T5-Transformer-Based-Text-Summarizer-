from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained("models").to(device)
tokenizer = T5Tokenizer.from_pretrained("models")


def clean_data(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def summarize_dialogue(dialogue):
    dialogue = clean_data(dialogue)
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(targets[0], skip_special_tokens=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/summarize")
async def summarize(request: Request):
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
        dialogue = data.get("dialogue", "")
    else:
        form = await request.form()
        dialogue = form.get("dialogue", "")

    return {"summary": summarize_dialogue(dialogue)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
