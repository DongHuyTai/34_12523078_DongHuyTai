# Local Server Integration and AI Deployment Strategy
## Topic: Setup, Testing, and Deployment of a Naive Bayes Model

## 1. Objective

* Run the API Service Server and a stable Naive Bayes AI server concurrently on the local machine.
* Fully own the source code, especially the training, probability calculation, and prediction pipeline of the Naive Bayes model.
* Use Docker to virtualize and synchronize the runtime environment between the local development environment and Production.
* Package the system, test the prediction endpoint, verify system health, and deploy a stable Naive Bayes model to Production.

## 2. Required Technologies

* Docker
* Naive Bayes Model (classification model)
* Model Training & Prediction
* API Service Server
* RESTful Endpoint
* Health Check Monitoring
* Command Line Testing Tools

## 3. Architecture

The system follows a Multi-component Architecture, in which the API Server receives input data, passes it to the Naive Bayes processing layer, and returns the predicted class along with its corresponding probability.

```text
Local Integration Suite

Local machine / Container:
 ├── Component 1 (API Endpoint)
 ├── Component 2 (Data Preprocessing)
 ├── Naive Bayes Model (Prior & Likelihood computation)
 └── Docker Environment (Virtualization layer)
```

API, preprocessing, and the model are kept as separate components so each can be tested independently.

## 4. Input

* **Request format:** JSON payload, file, or raw string, depending on the endpoint specification.
* **Primary field:** `features` — an array of feature values representing the data point to classify.
* **Example request body:**

```json
{
  "features": ["feature_1", "feature_2", "feature_3"]
}
```

* **Preprocessing requirement:** input data must pass through the preprocessing step (cleaning, encoding/normalization) before being validated and fed into the Naive Bayes model.
* **Test inputs:** command-line test cases used to validate the endpoint and model, including expected status codes, expected predicted class, and expected probability for comparison against actual results.

## 5. Output

* **Response format:** JSON.
* **Primary fields:**
  * `success` — whether the request completed successfully.
  * `status` — HTTP status code.
  * `message` — human-readable result message.
  * `data.model` — name of the model used (`naive_bayes`).
  * `data.endpoint` — the endpoint path that served the request.
  * `data.prediction` — the predicted class label.
  * `data.probability` — the posterior probability associated with the predicted class.
  * `data.health_status` — current health status of the model/service.

* **Example successful response:**

```json
{
  "success": true,
  "status": 200,
  "message": "Naive Bayes prediction successful",
  "data": {
    "model": "naive_bayes",
    "endpoint": "/api/v1/predict",
    "prediction": "class_1",
    "probability": 0.87,
    "health_status": "healthy"
  }
}
```

## 6. Algorithm (Naive Bayes) — Required

### 6.1 Overview

Naive Bayes is a probabilistic classifier based on Bayes' Theorem, under the "naive" assumption that all features are conditionally independent given the class label. The model computes the posterior probability of each class given the input features and selects the class with the highest posterior probability as the prediction.

### 6.2 Bayes' Theorem

For a class `C` and a feature vector `X = (x1, x2, ..., xn)`:

```
P(C | X) = [ P(X | C) * P(C) ] / P(X)
```

Where:
* `P(C | X)` — posterior probability of class `C` given features `X`.
* `P(X | C)` — likelihood of observing `X` given class `C`.
* `P(C)` — prior probability of class `C`.
* `P(X)` — evidence (marginal probability of `X`), constant across all classes and therefore omittable during comparison.

### 6.3 Naive (Conditional Independence) Assumption

Because features are assumed conditionally independent given the class:

```
P(X | C) = P(x1 | C) * P(x2 | C) * ... * P(xn | C)
```

So the posterior becomes proportional to:

```
P(C | X) ∝ P(C) * Π P(xi | C),  for i = 1 to n
```

### 6.4 Training Phase

1. **Input:** a normalized/preprocessed training dataset, where each record has a feature vector and a known class label.
2. **Compute Prior — `P(C)`** for each class:
   ```
   P(C) = (number of samples in class C) / (total number of samples)
   ```
3. **Compute Likelihood — `P(xi | C)`** for each feature value given each class:
   * For categorical/discrete features, use frequency counts (optionally with Laplace smoothing to avoid zero probabilities):
     ```
     P(xi | C) = (count of xi in class C + 1) / (total count in class C + number of possible values of xi)
     ```
   * For continuous features (Gaussian Naive Bayes), model the feature as normally distributed per class and use the Gaussian probability density function with the per-class mean and variance.
4. **Output of training:** the set of learned parameters — priors `P(C)` for every class and likelihoods `P(xi | C)` for every feature/class combination — persisted for use at prediction time.

### 6.5 Prediction Phase

1. **Input:** a new feature vector `X` (after preprocessing) from an incoming API request.
2. For each candidate class `C`, compute the (unnormalized) posterior score:
   ```
   score(C) = P(C) * Π P(xi | C),  for i = 1 to n
   ```
3. **Select the predicted class** as the class with the maximum score:
   ```
   prediction = argmax_C [ score(C) ]
   ```
4. **Compute the probability** to return to the client by normalizing the score across all classes:
   ```
   probability(C) = score(C) / Σ score(C'),  for all classes C'
   ```
5. **Return** the predicted class label and its normalized probability as the endpoint response.

### 6.6 End-to-End Flow

```
Incoming Request
      │
      ▼
Preprocessing (cleaning / encoding / normalization)
      │
      ▼
Naive Bayes Model
  ├── Load trained Prior P(C)
  ├── Load trained Likelihood P(xi | C)
  ├── Compute score(C) for each class
  └── Select prediction = argmax_C score(C)
      │
      ▼
Response: { prediction, probability, health_status }
```

## 7. Detailed Workflow

### 7.1 Component Design

* Build the first API Endpoint to receive data to be classified and test it independently.
* Separate the preprocessing step, the Naive Bayes computation, and the prediction result so each can be tested in isolation.
* Initialize the endpoint and define the path through which the API communicates with the model processing layer.

### 7.2 Parameter Testing

* Support input data as JSON, file, or raw string, depending on the endpoint specification.
* Validate the preprocessed input data before it is passed into the Naive Bayes model.
* Check the endpoint's response status code, and cross-check the predicted class and probability against the expected results.
* Run test commands directly from the command line to validate both the endpoint and the model.

### 7.3 Local AI Integration

* Train the Naive Bayes model on a normalized dataset and persist the parameters needed for prediction.
* On each request, the API Server performs preprocessing and then feeds the data into the Naive Bayes model.
* The model computes the posterior probability for each class and selects the highest-probability class as the prediction.
* Run the API Server and the Naive Bayes model locally to verify the full flow from input to prediction.

### 7.4 Dockerization & Production Deployment

* Use Docker to package the API Server, model source code, and required dependencies.
* Set up a Health Check to verify the API and confirm the Naive Bayes model is ready to serve predictions.
* Package the finished product and deploy to Production after successful testing.

## 8. Basic Docker Operations

Quick local test configuration:

```bash
# Start the API Server and Naive Bayes model with Docker
docker compose up --build

# Test the prediction endpoint
curl -X POST http://localhost:3000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features":["feature_1","feature_2","feature_3"]}'
```

## 9. Evaluation Criteria

| Category | Points |
|---|---|
| Component-separated architecture | 20 |
| Input preprocessing & testing | 20 |
| Naive Bayes training & prediction | 25 |
| Containerization & Docker configuration | 15 |
| Health Check & Production deployment | 10 |
| Code ownership mindset | 10 |

## 10. Submission Requirements

* Complete source code for the API and the Naive Bayes model.
* `Dockerfile` and `docker-compose.yml` configuration files.
* Documentation describing the input data, training process, prediction process, and endpoints.
* Command-line test scripts for the endpoint.
* Video/screenshot evidence of the system successfully running the Naive Bayes model.
