# Scaling Guide for Log Anomaly Detection

This guide shows how to scale the log anomaly detection system for production use with MCPs, AI platforms, and enterprise infrastructure.

## Table of Contents
1. [Quick Start](#quick-start)
2. [MCP Server Setup](#mcp-server-setup)
3. [REST API Deployment](#rest-api-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Batch Processing](#batch-processing)
6. [Integration Examples](#integration-examples)
7. [Production Considerations](#production-considerations)

---

## Quick Start

### Install API Dependencies
```bash
pip install -r requirements_api.txt
```

### Option 1: REST API (Recommended for most integrations)
```bash
# Start API server
python anomaly_api.py

# Server runs on http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Option 2: MCP Server (For Claude and AI tools)
```bash
python anomaly_mcp_server.py
```

### Option 3: Docker (Production deployment)
```bash
docker-compose up -d
```

---

## MCP Server Setup

### What is MCP?
Model Context Protocol (MCP) allows AI tools like Claude Desktop to access your anomaly detection capabilities as a service.

### Configuration

#### For Claude Desktop
Add to `claude_desktop_config.json`:

**Windows:**
```json
{
  "mcpServers": {
    "log-anomaly-detection": {
      "command": "python",
      "args": [
        "C:\\Users\\jimmy\\.local\\bin\\anomaly_mcp_server.py"
      ],
      "env": {
        "MODEL_DIR": "anomaly_outputs"
      }
    }
  }
}
```

**macOS/Linux:**
```json
{
  "mcpServers": {
    "log-anomaly-detection": {
      "command": "python3",
      "args": [
        "/path/to/anomaly_mcp_server.py"
      ],
      "env": {
        "MODEL_DIR": "anomaly_outputs"
      }
    }
  }
}
```

### Available MCP Tools

1. **load_anomaly_models** - Load trained detection models
   ```json
   {
     "model_dir": "anomaly_outputs"
   }
   ```

2. **analyze_logs** - Analyze log data for threats
   ```json
   {
     "log_data": "[{\"timestamp\": \"...\", \"user\": \"...\"}]",
     "format": "json"
   }
   ```

3. **analyze_log_file** - Analyze file by path
   ```json
   {
     "filepath": "logs/auth.json"
   }
   ```

4. **get_detection_stats** - Get model statistics
   ```json
   {}
   ```

### Using from Claude

After configuration, you can ask Claude:
```
"Analyze the logs in my logs folder for security threats"
"Load the anomaly detection models and check recent authentication logs"
"What threats were detected in auth.json?"
```

---

## REST API Deployment

### Local Development
```bash
python anomaly_api.py
# Access: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Load Models
```bash
curl -X POST http://localhost:8000/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_dir": "anomaly_outputs"}'
```

#### Analyze Logs (JSON)
```bash
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
```

#### Analyze File Upload
```bash
curl -X POST http://localhost:8000/analyze/file \
  -F "file=@logs/auth.json"
```

#### Get Statistics
```bash
curl http://localhost:8000/stats
```

### Python Client Example
```python
from integration_examples import AnomalyDetectionClient

client = AnomalyDetectionClient("http://localhost:8000")

# Check health
health = client.health_check()

# Load models
client.load_models()

# Analyze logs
logs = [
    {
        "timestamp": "2026-01-15T10:00:00Z",
        "user": "admin",
        "source_ip": "10.0.0.100",
        "event_type": "login",
        "action": "failed",
        "message": "Failed login"
    }
]

result = client.analyze_logs(logs)
print(f"Detected {result['anomalies_detected']} anomalies")
```

---

## Docker Deployment

### Build and Run
```bash
# Build image
docker build -t anomaly-detection .

# Run API
docker run -d -p 8000:8000 \
  -v $(pwd)/anomaly_outputs:/app/anomaly_outputs \
  -v $(pwd)/logs:/app/logs \
  --name anomaly-api \
  anomaly-detection
```

### Docker Compose (Multi-Service)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services include:
- **anomaly-api** - REST API on port 8000
- **anomaly-mcp** - MCP server for AI tools
- **anomaly-batch** - Scheduled batch processor

### Production Docker Compose

```yaml
version: '3.8'

services:
  anomaly-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./anomaly_outputs:/app/anomaly_outputs
      - ./logs:/app/logs
    environment:
      - WORKERS=4
      - LOG_LEVEL=info
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - anomaly-api
    restart: always
```

---

## Batch Processing

### Run Once
```bash
python batch_processor.py \
  --log-dir logs/ \
  --output-dir batch_outputs/ \
  --once
```

### Continuous Processing
```bash
python batch_processor.py \
  --log-dir logs/ \
  --output-dir batch_outputs/ \
  --interval 3600
```

### Scheduled with Cron
```bash
# Edit crontab
crontab -e

# Add entry (run every hour)
0 * * * * cd /path/to/app && python batch_processor.py --once
```

### Docker Batch Processing
```bash
docker run -d \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/batch_outputs:/app/batch_outputs \
  -v $(pwd)/anomaly_outputs:/app/anomaly_outputs \
  -e BATCH_INTERVAL=3600 \
  anomaly-detection \
  python batch_processor.py
```

---

## Integration Examples

### 1. Slack Alerts
```python
from integration_examples import SlackAlerter, AnomalyDetectionClient

client = AnomalyDetectionClient()
slack = SlackAlerter("https://hooks.slack.com/services/YOUR/WEBHOOK/URL")

result = client.analyze_logs(logs)

for anomaly in result['anomalies']:
    if anomaly['severity'] in ['high', 'critical']:
        slack.send_alert(anomaly)
```

### 2. Splunk Integration
```python
from integration_examples import SplunkIntegration, AnomalyDetectionClient

client = AnomalyDetectionClient()
splunk = SplunkIntegration(
    hec_url="https://splunk.example.com:8088/services/collector",
    token="YOUR-HEC-TOKEN"
)

result = client.analyze_logs(logs)

for anomaly in result['anomalies']:
    splunk.send_event(anomaly)
```

### 3. Elasticsearch Indexing
```python
from integration_examples import ElasticsearchIntegration, AnomalyDetectionClient

client = AnomalyDetectionClient()
es = ElasticsearchIntegration("http://localhost:9200")

result = client.analyze_logs(logs)
es.bulk_index(result['anomalies'])
```

### 4. Webhook Alerts
```python
from integration_examples import WebhookAlerter, AnomalyDetectionClient

client = AnomalyDetectionClient()
webhook = WebhookAlerter("https://your-webhook.com/alerts")

result = client.analyze_logs(logs)

for anomaly in result['anomalies']:
    webhook.send_alert(anomaly)
```

### 5. Kafka Streaming
```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

client = AnomalyDetectionClient()
result = client.analyze_logs(logs)

for anomaly in result['anomalies']:
    producer.send('security-anomalies', anomaly)
```

---

## Production Considerations

### Performance Optimization

#### 1. Use Multiple Workers
```bash
uvicorn anomaly_api:app --workers 4 --host 0.0.0.0 --port 8000
```

#### 2. Enable Caching
Add Redis caching for repeated queries:
```python
import redis
from functools import lru_cache

cache = redis.Redis(host='localhost', port=6379, db=0)

@lru_cache(maxsize=1000)
def get_cached_analysis(log_hash):
    # Cache analysis results
    pass
```

#### 3. Batch Processing
Process logs in batches rather than one-by-one:
```python
# Process 1000 events at a time
batch_size = 1000
for i in range(0, len(logs), batch_size):
    batch = logs[i:i+batch_size]
    result = client.analyze_logs(batch)
```

### Security

#### 1. Add API Authentication
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/analyze")
async def analyze_logs(
    request: AnalysisRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Validate token
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401)
    # ... rest of code
```

#### 2. Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/analyze")
@limiter.limit("100/minute")
async def analyze_logs(request: AnalysisRequest):
    # ... code
```

#### 3. HTTPS/TLS
Use nginx reverse proxy with SSL:
```nginx
server {
    listen 443 ssl;
    server_name anomaly-api.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Monitoring

#### 1. Prometheus Metrics
```python
from prometheus_client import Counter, Histogram

anomalies_detected = Counter('anomalies_detected_total', 'Total anomalies detected')
analysis_duration = Histogram('analysis_duration_seconds', 'Analysis duration')

@analysis_duration.time()
def analyze():
    # ... analysis code
    anomalies_detected.inc(count)
```

#### 2. Health Checks
```python
@app.get("/health/live")
async def liveness():
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    if not MODEL_STATE["loaded"]:
        raise HTTPException(status_code=503)
    return {"status": "ready"}
```

### Scaling Horizontally

#### Load Balancer Setup
```nginx
upstream anomaly_backends {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
    server localhost:8004;
}

server {
    listen 80;
    location / {
        proxy_pass http://anomaly_backends;
    }
}
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-detection
spec:
  replicas: 4
  selector:
    matchLabels:
      app: anomaly-detection
  template:
    metadata:
      labels:
        app: anomaly-detection
    spec:
      containers:
      - name: api
        image: anomaly-detection:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "1"
            memory: "2Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: anomaly-detection
spec:
  selector:
    app: anomaly-detection
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Summary

You now have multiple deployment options:

- ✅ **MCP Server** - For Claude and AI platforms
- ✅ **REST API** - For any HTTP client
- ✅ **Docker** - For containerized deployment
- ✅ **Batch Processing** - For scheduled analysis
- ✅ **Integrations** - Slack, Splunk, Elasticsearch, Kafka, etc.

Choose the method that best fits your infrastructure and use case!
