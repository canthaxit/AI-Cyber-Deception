# REST API Service

FastAPI-based REST API for log anomaly detection.

## Files

- **anomaly_api.py** - Standard REST API server
- **anomaly_api_chronicle.py** - API with Google Chronicle integration
- **test_api.py** - API testing script
- **start_api.bat** - Quick start script (Windows)

## Quick Start

```bash
# Start standard API
python anomaly_api.py

# Or start API with Chronicle integration
python anomaly_api_chronicle.py

# API available at:
# - Main: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - Health: http://localhost:8000/health
```

## Test the API

```bash
# Start server
python anomaly_api.py &

# Run tests
python test_api.py
```

## API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /models/info` - Model information
- `GET /stats` - System statistics

### Models
- `POST /models/load` - Load trained models

### Analysis
- `POST /analyze` - Analyze JSON logs
- `POST /analyze/file` - Upload and analyze file

### Chronicle (if using anomaly_api_chronicle.py)
- `GET /chronicle/status` - Chronicle integration status
- `POST /chronicle/enable` - Enable Chronicle integration
- `POST /chronicle/disable` - Disable Chronicle integration
- `POST /chronicle/test` - Test Chronicle connection

## Example Usage

```bash
# Load models
curl -X POST http://localhost:8000/models/load

# Analyze logs
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      {
        "timestamp": "2026-01-15T10:00:00Z",
        "user": "admin",
        "source_ip": "10.0.0.100",
        "event_type": "login",
        "action": "failed",
        "message": "Failed login attempt"
      }
    ]
  }'

# Upload file for analysis
curl -X POST http://localhost:8000/analyze/file \
  -F "file=@../tests/test_logs_attack.json"
```

## Python Client

```python
import requests

base_url = "http://localhost:8000"

# Load models
requests.post(f"{base_url}/models/load",
              params={"model_dir": "../core/anomaly_outputs"})

# Analyze logs
response = requests.post(
    f"{base_url}/analyze",
    json={
        "logs": [{
            "timestamp": "2026-01-15T10:00:00Z",
            "user": "admin",
            "source_ip": "10.0.0.100",
            "event_type": "login",
            "action": "failed",
            "message": "Failed login"
        }]
    }
)

result = response.json()
print(f"Detected {result['anomalies_detected']} threats")
```

## Configuration

Set model directory:
```python
# In anomaly_api.py, update:
MODEL_DIR = "../core/anomaly_outputs"
```

Or use environment variable:
```bash
export MODEL_DIR=/path/to/anomaly-detection/core/anomaly_outputs
python anomaly_api.py
```

## Production Deployment

```bash
# Multiple workers
uvicorn anomaly_api:app --workers 4 --host 0.0.0.0 --port 8000

# With Chronicle
uvicorn anomaly_api_chronicle:app --workers 4 --host 0.0.0.0 --port 8000
```

## Documentation

See:
- [`../docs/README_SCALING.md`](../docs/README_SCALING.md) - API deployment guide
- [`../docs/SCALING_GUIDE.md`](../docs/SCALING_GUIDE.md) - Production scaling
- Interactive docs at http://localhost:8000/docs (when running)
