# Sentiment Analysis API

A REST API that classifies text as **positive** or **negative**, built with FastAPI and a quantized DistilBERT sentiment analysis model running on ONNX Runtime.

## Tech Stack

- **Python** — core language, isolated via `venv`
- **FastAPI** — web framework for the REST API
- **Uvicorn** — ASGI server that runs the FastAPI app
- **HuggingFace Transformers** — tokenizer and model config utilities for a pretrained sentiment model (`distilbert-base-uncased-finetuned-sst-2-english`)
- **ONNX Runtime** — runs the model's inference graph from a pre-exported, int8-quantized ONNX file (no PyTorch/TensorFlow at runtime)
- **Docker** — containerization for consistent deployment
- **Render** — hosting/deployment platform (free tier, 512MB RAM)

> **Why ONNX Runtime instead of PyTorch?** The original PyTorch-based implementation exceeded Render's 512MB free-tier memory limit. Swapping to a pre-quantized ONNX build of the same DistilBERT model cut resident memory to roughly 100MB, with no meaningful change in prediction accuracy.

## API Endpoints

### `GET /`

Health check.

**Response**

```json
{
  "status": "ok"
}
```

### `POST /predict`

Classifies the sentiment of a piece of text.

**Request**

```json
{
  "text": "I absolutely love this API!"
}
```

**Response**

```json
{
  "label": "POSITIVE",
  "score": 0.9998716115951538
}
```

`label` is either `"POSITIVE"` or `"NEGATIVE"`. `score` is the model's confidence, between 0 and 1.

## Running Locally

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs are auto-generated at `http://127.0.0.1:8000/docs`.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is terrible and I hate it."}'
```

## Running with Docker

Build the image:

```bash
docker build -t sentiment-analysis-api .
```

Run the container:

```bash
docker run -p 8000:8000 sentiment-analysis-api
```

The API will be available at `http://localhost:8000`.

## Deployment

This project is deployed to [Render](https://render.com) as a Dockerized web service.
