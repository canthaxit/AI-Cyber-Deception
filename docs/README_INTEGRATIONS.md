# Log Anomaly Detection - Complete Integration Guide

Enterprise-ready log anomaly detection with support for all major SIEM platforms, AI tools, and cloud services.

## 🚀 Quick Start

Choose your integration method:

| Method | Use Case | Setup Time | Guide |
|--------|----------|------------|-------|
| **Google Chronicle** | Google Cloud SIEM | 5 min | [CHRONICLE_QUICK_START.md](CHRONICLE_QUICK_START.md) |
| **REST API** | Any platform via HTTP | 2 min | [README_SCALING.md](README_SCALING.md) |
| **MCP Server** | Claude, AI platforms | 3 min | [claude_mcp_config.json](claude_mcp_config.json) |
| **Direct Python** | Custom integration | 1 min | [integration_examples.py](integration_examples.py) |
| **Docker** | Production deployment | 5 min | [docker-compose.yml](docker-compose.yml) |

## 📋 What's Included

### Core Detection System
- ✅ **Unsupervised anomaly detection** (no labeled data needed)
- ✅ **Multi-model ensemble** (Isolation Forest + Statistical rules)
- ✅ **Real-time and batch processing**
- ✅ **Automatic threat classification** (brute force, privilege escalation, etc.)
- ✅ **Severity scoring** (low, medium, high, critical)

### Integration Options

#### SIEM Platforms
- **Google Chronicle** - Full integration with UDM format and detection rules
- **Splunk** - HTTP Event Collector (HEC) integration
- **Elasticsearch** - Direct indexing with bulk API
- **Generic SIEM** - Syslog protocol support

#### AI & Automation
- **MCP Server** - For Claude Desktop and AI platforms
- **REST API** - OpenAPI/Swagger documented endpoints
- **Webhooks** - Real-time alert forwarding
- **Slack** - Formatted notifications with severity colors

#### Cloud & Streaming
- **Kafka** - Event streaming integration
- **AWS Lambda** - Serverless deployment
- **Azure Functions** - Container app deployment
- **Batch Processing** - Scheduled cron jobs

## 🎯 Choose Your Integration

### Option 1: Google Chronicle SIEM ⭐ Recommended for Google Cloud

**Perfect for:**
- Google Cloud environments
- Teams using Chronicle/Security Operations
- Enterprise security operations centers

**Features:**
- Automatic anomaly forwarding
- UDM (Unified Data Model) conversion
- Custom detection rules (YARA-L)
- Real-time alerting
- Background processing (non-blocking)

**Quick Setup:**
```bash
# Run automated setup
setup_chronicle.bat

# Or follow manual steps
pip install -r requirements_chronicle.txt
python google_chronicle_integration.py --setup
python google_chronicle_integration.py --test
```

**Start API with Chronicle:**
```bash
python anomaly_api_chronicle.py
# Automatically sends all anomalies to Chronicle
```

**See detailed guide:** [GOOGLE_SIEM_SETUP.md](GOOGLE_SIEM_SETUP.md)

---

### Option 2: REST API 🌐 Universal Integration

**Perfect for:**
- Any platform with HTTP client
- Microservices architecture
- Multi-platform environments

**Features:**
- RESTful JSON API
- Auto-generated docs (OpenAPI/Swagger)
- CORS enabled
- Background processing
- Rate limiting ready

**Quick Setup:**
```bash
# Windows
start_api.bat

# Linux/Mac
python anomaly_api.py
```

**API Docs:** http://localhost:8000/docs

**Example usage:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"logs": [...]}'
```

**See detailed guide:** [README_SCALING.md](README_SCALING.md)

---

### Option 3: MCP Server 🤖 AI Platform Integration

**Perfect for:**
- Claude Desktop users
- AI-assisted security analysis
- Natural language log queries

**Features:**
- Model Context Protocol support
- Natural language interaction
- Automatic model loading
- File and data analysis

**Quick Setup:**
```bash
# Copy MCP config to Claude Desktop
copy claude_mcp_config.json "%APPDATA%\Claude\claude_desktop_config.json"

# Restart Claude Desktop
```

**Use with Claude:**
```
"Analyze the logs in my logs folder for security threats"
"Load anomaly detection models and check auth.json for brute force attacks"
"What high-severity anomalies were detected today?"
```

**See detailed guide:** [README_SCALING.md](README_SCALING.md#mcp-integration-claude--ai-tools)

---

### Option 4: Direct Python Import 🐍 Custom Integration

**Perfect for:**
- Custom Python applications
- Data science workflows
- Embedded security modules

**Quick Setup:**
```python
from log_anomaly_detection_lite import (
    LogParser, LogFeaturePipeline,
    StatisticalAnomalyDetector, AnomalyScorer
)
from sklearn.ensemble import IsolationForest
import joblib

# Load models
feature_pipeline = joblib.load("anomaly_outputs/feature_pipeline.pkl")
iso_forest = joblib.load("anomaly_outputs/isolation_forest_model.pkl")
stat_detector = joblib.load("anomaly_outputs/statistical_detector.pkl")

# Analyze logs
parser = LogParser()
df = parser.load_logs("logs/")
features = feature_pipeline.transform(df)
iso_scores = -iso_forest.score_samples(features)
stat_scores = stat_detector.detect_all(df)

# Get anomalies
scorer = AnomalyScorer()
combined_scores = scorer.combine_scores({
    'isolation_forest': iso_scores,
    'statistical': stat_scores
})

anomalies = df[combined_scores > 0.7]
```

**See examples:** [integration_examples.py](integration_examples.py)

---

### Option 5: Docker Deployment 🐳 Production Ready

**Perfect for:**
- Production environments
- Kubernetes clusters
- Cloud deployments

**Quick Setup:**
```bash
# Single container
docker build -t anomaly-detection .
docker run -d -p 8000:8000 anomaly-detection

# Multi-service
docker-compose up -d
```

**Services included:**
- **anomaly-api** - REST API server
- **anomaly-mcp** - MCP server for AI tools
- **anomaly-batch** - Scheduled batch processor

**See detailed guide:** [README_SCALING.md](README_SCALING.md#docker-deployment)

---

## 🔗 SIEM Platform Integrations

### Google Chronicle

```python
from google_chronicle_integration import ChronicleClient

chronicle = ChronicleClient(
    credentials_file="chronicle_credentials.json",
    customer_id="C00000000",
    region="us"
)

chronicle.send_anomalies(anomalies)
```

**Guide:** [GOOGLE_SIEM_SETUP.md](GOOGLE_SIEM_SETUP.md)

### Splunk

```python
from integration_examples import SplunkIntegration

splunk = SplunkIntegration(
    hec_url="https://splunk.example.com:8088/services/collector",
    token="YOUR-HEC-TOKEN"
)

for anomaly in anomalies:
    splunk.send_event(anomaly)
```

### Elasticsearch

```python
from integration_examples import ElasticsearchIntegration

es = ElasticsearchIntegration("http://localhost:9200")
es.bulk_index(anomalies)
```

### Generic SIEM (Syslog)

```python
from integration_examples import SIEMIntegration

siem = SIEMIntegration(syslog_host="siem.example.com", syslog_port=514)

for anomaly in anomalies:
    siem.send_syslog(anomaly)
```

---

## 📊 Alert & Notification Integrations

### Slack

```python
from integration_examples import SlackAlerter

slack = SlackAlerter("https://hooks.slack.com/services/YOUR/WEBHOOK/URL")
slack.send_alert(anomaly)
```

### Webhooks

```python
from integration_examples import WebhookAlerter

webhook = WebhookAlerter("https://your-webhook.com/alerts")
webhook.send_alert(anomaly)
```

### Email (via SMTP)

```python
import smtplib
from email.mime.text import MIMEText

msg = MIMEText(json.dumps(anomaly, indent=2))
msg['Subject'] = f"Security Alert: {anomaly['threat_type']}"
msg['From'] = "alerts@example.com"
msg['To'] = "security@example.com"

smtp = smtplib.SMTP('localhost')
smtp.send_message(msg)
```

---

## 🌊 Streaming Integrations

### Kafka

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

for anomaly in anomalies:
    producer.send('security-anomalies', anomaly)
```

### AWS Kinesis

```python
import boto3
import json

kinesis = boto3.client('kinesis')

for anomaly in anomalies:
    kinesis.put_record(
        StreamName='security-anomalies',
        Data=json.dumps(anomaly),
        PartitionKey=anomaly['user']
    )
```

---

## ☁️ Cloud Platform Integrations

### AWS Lambda

```python
def lambda_handler(event, context):
    from anomaly_api import AnomalyDetectionClient

    client = AnomalyDetectionClient()
    logs = json.loads(event['body'])['logs']
    result = client.analyze_logs(logs)

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### Azure Functions

```python
import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
    from anomaly_api import AnomalyDetectionClient

    client = AnomalyDetectionClient()
    logs = req.get_json()['logs']
    result = client.analyze_logs(logs)

    return func.HttpResponse(
        json.dumps(result),
        mimetype="application/json"
    )
```

### Google Cloud Run

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements_api.txt
CMD ["python", "anomaly_api.py"]
```

Deploy:
```bash
gcloud run deploy anomaly-detection --source . --platform managed
```

---

## 📈 Monitoring & Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

anomalies_detected = Counter('anomalies_detected_total', 'Total anomalies')
analysis_duration = Histogram('analysis_duration_seconds', 'Analysis time')

@analysis_duration.time()
def analyze():
    # ... analysis
    anomalies_detected.inc(count)
```

### Grafana Dashboard

Connect to API metrics endpoint:
```bash
curl http://localhost:8000/metrics
```

### Health Checks

```bash
# Liveness
curl http://localhost:8000/health/live

# Readiness
curl http://localhost:8000/health/ready
```

---

## 🔒 Security Best Practices

1. **API Authentication**
   - Use API keys or JWT tokens
   - Enable HTTPS/TLS
   - Implement rate limiting

2. **Credential Management**
   - Store credentials in secrets manager
   - Use environment variables
   - Rotate keys regularly

3. **Network Security**
   - Use VPC/private networks
   - Enable firewall rules
   - Restrict IP access

4. **Audit Logging**
   - Log all API access
   - Monitor for unusual patterns
   - Alert on failed authentications

---

## 📚 Complete Documentation

| Document | Description |
|----------|-------------|
| [CHRONICLE_QUICK_START.md](CHRONICLE_QUICK_START.md) | 5-minute Google Chronicle setup |
| [GOOGLE_SIEM_SETUP.md](GOOGLE_SIEM_SETUP.md) | Complete Chronicle integration guide |
| [README_SCALING.md](README_SCALING.md) | API, MCP, and Docker deployment |
| [SCALING_GUIDE.md](SCALING_GUIDE.md) | Production deployment guide |
| [integration_examples.py](integration_examples.py) | Code examples for all platforms |

---

## 🎓 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Log Sources                              │
│  Files • Streams • APIs • Cloud Logs • SIEM Forwarders      │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐           ┌───────▼────────┐
│   REST API     │           │  MCP Server    │
│   (HTTP)       │           │  (AI Tools)    │
└───────┬────────┘           └───────┬────────┘
        │                             │
        └──────────────┬──────────────┘
                       │
              ┌────────▼─────────┐
              │ Detection Engine │
              │ • Isolation      │
              │ • Statistical    │
              │ • Features       │
              └────────┬─────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼───────┐ ┌───▼──────┐ ┌────▼─────┐
│  Chronicle    │ │  Splunk  │ │  Slack   │
│  (Google)     │ │  (HEC)   │ │ (Webhook)│
└───────────────┘ └──────────┘ └──────────┘
        │              │              │
┌───────▼───────┐ ┌───▼──────┐ ┌────▼─────┐
│ Elasticsearch │ │  Kafka   │ │  Custom  │
└───────────────┘ └──────────┘ └──────────┘
```

---

## 🆘 Support & Troubleshooting

### Common Issues

**Models not loading:**
```bash
# Retrain models
python log_anomaly_detection_lite.py --data_path test_logs/
```

**API not starting:**
```bash
# Check port availability
netstat -an | grep 8000

# Use different port
python anomaly_api.py --port 8001
```

**Chronicle not connecting:**
```bash
# Test credentials
python google_chronicle_integration.py --test

# Check permissions
gcloud projects get-iam-policy PROJECT_ID
```

### Get Help

1. Check the relevant documentation above
2. Review [integration_examples.py](integration_examples.py) for code samples
3. Test with sample data first
4. Enable debug logging

---

## 🎯 Choose Your Path

| If you want to... | Use this... |
|-------------------|-------------|
| **Integrate with Google Cloud** | [Google Chronicle Setup](CHRONICLE_QUICK_START.md) |
| **Build custom integration** | [REST API](README_SCALING.md) |
| **Use with Claude/AI tools** | [MCP Server](README_SCALING.md#mcp-integration-claude--ai-tools) |
| **Deploy to production** | [Docker Guide](README_SCALING.md#docker-deployment) |
| **See code examples** | [integration_examples.py](integration_examples.py) |

---

**Ready to detect threats at enterprise scale!** 🛡️
