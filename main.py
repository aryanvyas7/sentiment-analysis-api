import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer

app = FastAPI(title="Sentiment Analysis API")

MODEL_ID = "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
ONNX_FILE = "onnx/model_quantized.onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
config = AutoConfig.from_pretrained(MODEL_ID)

onnx_model_path = hf_hub_download(repo_id=MODEL_ID, filename=ONNX_FILE)
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
session_input_names = {inp.name for inp in session.get_inputs()}


class TextRequest(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="np")
    onnx_inputs = {name: inputs[name] for name in session_input_names if name in inputs}

    logits = session.run(None, onnx_inputs)[0]
    scores = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    predicted_id = int(np.argmax(scores, axis=-1)[0])
    return {
        "label": config.id2label[predicted_id],
        "score": float(scores[0][predicted_id]),
    }
