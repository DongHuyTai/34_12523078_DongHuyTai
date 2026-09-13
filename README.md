# Local Server AI Strategy - Naive Bayes

This project provides a compact local deployment of a Naive Bayes classifier behind a FastAPI service. The API accepts a JSON payload with a list of feature values, preprocesses it, and returns the predicted class and posterior probability.

## Features

- Separate preprocessing and prediction logic
- Naive Bayes implementation with Laplace smoothing
- REST endpoint for prediction and health checks
- Docker support for local and production-like deployment
- Basic command-line validation scripts

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 3000
```

## Test the API

### Correct classification examples

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Please submit the project report before Friday"}'
```

Expected response:

```json
{
  "success": true,
  "status": 200,
  "message": "Naive Bayes prediction successful",
  "data": {
    "model": "naive_bayes",
    "endpoint": "/api/v1/predict",
    "prediction": "work",
    "probability": 0.87,
    "health_status": "healthy"
  }
}
```

Other valid examples:

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Hi mom, let\'s have dinner this weekend"}'
```

Expected prediction: `personal`

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Congratulations! You have won a free prize. Click now."}'
```

Expected prediction: `spam`

### Incorrect / invalid examples

These inputs should be rejected or return the wrong category if the classification logic is not followed.

#### 1. Empty input

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":""}'
```

Expected:

```json
{
  "detail": "features cannot be empty"
}
```

#### 2. Wrong field type

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[123,456]}'
```

Expected:

```json
{
  "detail": "all feature values must be strings"
}
```

#### 3. Wrong prediction should not pass

```bash
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Free money click here now winner"}'
```

Expected correct label: `spam`

This case is considered incorrect if the API returns `work` or `personal`.

## Docker

```bash
docker compose up --build
```

## Endpoint summary

- `GET /health` — service health status
- `POST /api/v1/predict` — classify an email text or feature vector

## Test case summary

### Correct test cases

| No. | Input text | Expected prediction | Expected probability |
|---|---|---|---|
| 1 | "Please submit the project report before Friday" | work | 0.87 |
| 2 | "Hi mom, let's have dinner this weekend" | personal | near 0.80 |
| 3 | "Congratulations! You have won a free prize. Click now." | spam | near 0.80 |

### Incorrect test cases

| No. | Input text | Expected result |
|---|---|---|
| 1 | "" | 400 error: features cannot be empty |
| 2 | {"features":[123,456]} | 400 error: all feature values must be strings |
| 3 | "Free money click here now winner" | should be spam, not work/personal |

## Model notes

The implementation uses a categorical Naive Bayes model with Laplace smoothing. Prior probabilities are computed from the class distribution in the training dataset, while likelihoods are estimated via feature value counts for each class.
