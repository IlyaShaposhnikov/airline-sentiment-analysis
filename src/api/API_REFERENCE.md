# API Reference

[![API Version](https://img.shields.io/badge/API-v1.0.0-blue)](https://github.com/IlyaShaposhnikov/airline-sentiment-analysis/releases)

Comprehensive reference for the Airline Sentiment Analysis REST API endpoints.

> **Base URL**: `http://localhost:8000` (configurable via `serving.api.host` and `serving.api.port`)

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs  
  Full interactive interface with live validation, request examples, and "Execute" buttons.

- **ReDoc**: http://localhost:8000/redoc  
  Alternative, highly readable documentation layout.

- **OpenAPI Schema**: http://localhost:8000/openapi.json  
  Raw JSON schema for integration with external tools and code generators.

## Authentication

The API currently runs without authentication. For production deployment, consider adding:
- API keys via `X-API-Key` header
- JWT tokens for user sessions
- Rate limiting via `serving.api.rate_limit_per_minute`

## Endpoints

### `GET /` — Root Endpoint

Service metadata and available endpoint list.

**Response (200 OK)**:
```json
{
  "service": "Airline Sentiment Analysis API",
  "version": "1.0.0",
  "description": "REST API for sentiment classification of airline tweets",
  "documentation": {
    "swagger_ui": "/docs",
    "redoc": "/redoc"
  },
  "endpoints": {
    "GET /": "This endpoint — API information",
    "GET /health": "Health check with model status",
    "POST /predict": "Single text sentiment prediction",
    "POST /predict/batch": "Batch prediction (1-100 texts)"
  },
  "config": {
    "host": "0.0.0.0",
    "port": 8000,
    "enabled": true
  }
}
```

---

### `GET /health` — Health Check

Service readiness probe for load balancers and monitoring systems.

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "service": "airline-sentiment-api",
  "timestamp": "2026-05-26T10:30:00.123456+00:00",
  "model_loaded": true,
  "shap_available": false
}
```

**Response (503 Service Unavailable)**:
```json
{
  "error": "service_unavailable",
  "detail": "API is disabled in configuration",
  "timestamp": "2026-05-26T10:30:00.123456+00:00"
}
```

---

### `POST /predict` — Single Prediction

Predict sentiment for a single text input.

**Request**:
```json
{
  "text": "Great flight with @VirginAmerica, excellent service!",
  "explain": true,
  "use_shap": false,
  "n_explain": 5
}
```

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `text` | string | ✅ | Input text for analysis | 1–1000 chars, non-empty |
| `explain` | boolean | ❌ | Include word-level explanation | Default `false` |
| `use_shap` | boolean | ❌ | Use SHAP instead of model weights | Requires `shap` package |
| `n_explain` | integer | ❌ | Number of top words in explanation | 1–20, default `5` |

**Response (200 OK)**:
```json
{
  "text": "Great flight with @VirginAmerica, excellent service!",
  "predicted_class": "positive",
  "predicted_class_idx": 1,
  "probabilities": {
    "negative": 0.03,
    "positive": 0.94,
    "neutral": 0.03
  },
  "confidence": 0.94,
  "timestamp": "2026-05-26T10:30:00.123456+00:00",
  "explanation": {
    "method": "weights",
    "top_contributors": [
      ["great", 1.85],
      ["excellent", 1.42],
      ["flight", 0.67]
    ]
  }
}
```

**Errors**:
| Status | Cause | Example Response |
|--------|-------|------------------|
| `422` | Schema validation failed (Pydantic) | `{"error": "validation_error", "detail": "Request validation failed"}` |
| `500` | Internal prediction error | `{"error": "internal_error", "detail": "Prediction failed: RuntimeError"}` |
| `504` | Prediction timeout (>30s) | `{"error": "timeout", "detail": "Prediction timed out after 30s"}` |

---

### `POST /predict/batch` — Batch Prediction

Predict sentiment for multiple texts in a single request.

**Request**:
```json
{
  "texts": [
    "Great flight!",
    "Terrible delay, never again",
    "Meh, okay experience"
  ],
  "explain": false,
  "use_shap": false,
  "n_explain": 5
}
```

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `texts` | array[string] | ✅ | List of texts to analyze | 1–100 items, each 1–1000 chars |
| `explain` | boolean | ❌ | Include explanations for all items | Default `false` |
| `use_shap` | boolean | ❌ | Use SHAP explanations | Requires `shap` |
| `n_explain` | integer | ❌ | Top words per explanation | 1–20 |

**Response (200 OK)**:
```json
{
  "status": "success",
  "count": 3,
  "predictions": [
    {
      "text": "Great flight!",
      "predicted_class": "positive",
      "predicted_class_idx": 1,
      "probabilities": {"negative": 0.05, "positive": 0.92, "neutral": 0.03},
      "confidence": 0.92,
      "timestamp": "2026-05-26T10:30:00.123456+00:00"
    },
    {
      "text": "Terrible delay, never again",
      "predicted_class": "negative",
      "predicted_class_idx": 0,
      "probabilities": {"negative": 0.88, "positive": 0.04, "neutral": 0.08},
      "confidence": 0.88,
      "timestamp": "2026-05-26T10:30:00.123457+00:00"
    },
    {
      "text": "Meh, okay experience",
      "predicted_class": "neutral",
      "predicted_class_idx": 2,
      "probabilities": {"negative": 0.15, "positive": 0.20, "neutral": 0.65},
      "confidence": 0.65,
      "timestamp": "2026-05-26T10:30:00.123458+00:00"
    }
  ],
  "processing_time_ms": 142.5
}
```

> **Partial Success**: If individual items fail during processing, they are logged and skipped. The `count` field reflects the number of **successful** predictions returned.
> **Timeout**: Batch requests have a scalable timeout: `30 seconds × number of texts`. For 10 texts, the total timeout is ~5 minutes.

**Errors**:
| Status | Cause | Example Response |
|--------|-------|------------------|
| `422` | Schema validation failed (Pydantic) | `{"error": "validation_error", "detail": "Request validation failed"}` |
| `500` | Internal batch error | `{"error": "internal_error", "detail": "Batch prediction failed"}` |
| `504` | Batch timeout (>30s × N texts) | `{"error": "timeout", "detail": "Batch prediction timed out"}` |

## Data Schemas (Pydantic v2)

### `PredictionRequest`
```python
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    explain: bool = False
    use_shap: bool = False
    n_explain: int = Field(5, ge=1, le=20)
```

### `PredictionResponse`
```python
class PredictionResponse(BaseModel):
    text: str
    predicted_class: str
    predicted_class_idx: int
    probabilities: Dict[str, float]
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime  # UTC
    explanation: Optional[Explanation] = None

class Explanation(BaseModel):
    method: Literal["weights", "shap"]
    top_contributors: List[Tuple[str, float]]
```

### `ErrorResponse` (Standard Error Format)
```python
class ErrorResponse(BaseModel):
    error: str  # Error code: "validation_error", "timeout", "internal_error"
    detail: str  # Human-readable description
    timestamp: datetime  # UTC
```

## Usage Examples

### curl — Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Amazing service, will fly again!",
    "explain": true,
    "n_explain": 3
  }'
```

### Python requests — Batch Prediction
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "texts": ["Love it!", "Hate it", "It's okay"],
        "explain": False
    },
    timeout=30
)

if response.status_code == 200:
    results = response.json()
    for pred in results["predictions"]:
        print(f"{pred['text'][:30]}... → {pred['predicted_class']} ({pred['confidence']:.2f})")
```

### JavaScript fetch — With Error Handling
```javascript
async function predictSentiment(text) {
  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, explain: true })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(`${response.status}: ${error.detail}`);
    }
    
    const result = await response.json();
    console.log(`Prediction: ${result.predicted_class} (${result.confidence})`);
    return result;
  } catch (err) {
    console.error('Prediction failed:', err.message);
    throw err;
  }
}
```

## Deployment & Configuration

### Local Run
```bash
# 1. Ensure model is trained
python scripts/train.py

# 2. Start API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Verify health endpoint
curl http://localhost:8000/health
```

### Docker (Example)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build & run
docker build -t airline-sentiment-api .
docker run -p 8000:8000 -e MODEL_PATH=/models/model_bundle.joblib airline-sentiment-api
```

### Production Environment Variables
```bash
# .env.example
MODEL_PATH=/mnt/models/v1/model_bundle.joblib
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=WARNING
CORS_ORIGINS=["https://myapp.com", "https://admin.myapp.com"]
```

## FAQ

**Q: Why can `confidence` be < 1.0 even for "obvious" texts?**  
A: The model outputs calibrated probabilities based on learned weights. Even strong signals typically yield 0.92–0.98 due to regularization and training noise.

**Q: What does `method: "weights"` mean in explanations?**  
A: Explanations use raw logistic regression coefficients. If `method: "shap"`, SHAP values were computed (requires the `shap` package).

**Q: Why does a batch return fewer results than requested?**  
A: Some items may have failed during processing (e.g., empty text after cleaning). Errors are logged, and successful predictions are returned. Check server logs for details.

**Q: How do I change the API log level?**  
A: Set `serving.api.log_level: "WARNING"` in `config.yaml`, or override via environment variable: `API_LOG_LEVEL=ERROR`. Available levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`.