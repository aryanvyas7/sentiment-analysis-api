# Sentiment Analysis API

A REST API that classifies text as **positive** or **negative**, built with FastAPI and a HuggingFace Transformers sentiment analysis model.

## Tech Stack

- **Python** — core language, isolated via `venv`
- **FastAPI** — web framework for the REST API
- **Uvicorn** — ASGI server that runs the FastAPI app
- **HuggingFace Transformers** — pretrained sentiment analysis model (`distilbert-base-uncased-finetuned-sst-2-english`)
- **PyTorch** — deep learning backend the model runs on
- **Docker** — containerization for consistent deployment
- **Render** — hosting/deployment platform

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
  "score": 0.999871015548706
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
