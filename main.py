from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment Analysis API")

sentiment_pipeline = pipeline("sentiment-analysis")


class TextRequest(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: TextRequest):
    result = sentiment_pipeline(request.text)[0]
    return {"label": result["label"], "score": result["score"]}
