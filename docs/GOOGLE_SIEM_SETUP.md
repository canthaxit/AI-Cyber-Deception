# Google Chronicle SIEM Integration Guide

Complete guide for integrating log anomaly detection with Google Security Operations (Chronicle).

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup Steps](#setup-steps)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Detection Rules](#detection-rules)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This integration sends detected security anomalies to Google Chronicle SIEM in real-time using the Chronicle Ingestion API.

**Features:**
- ✅ Automatic anomaly forwarding to Chronicle
- ✅ UDM (Unified Data Model) format conversion
- ✅ Custom detection rules in Chronicle
- ✅ Batch and real-time ingestion
- ✅ Background processing (non-blocking)
- ✅ Full REST API integration

**Architecture:**
```
Logs → Anomaly Detection → Chronicle API → Chronicle SIEM
                                 ↓
                        Detection Rules → Alerts
```

---

## Prerequisites

### 1. Google Cloud Project

You need a Google Cloud project with Chronicle enabled:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Chronicle API:
   ```bash
   gcloud services enable chronicle.googleapis.com
   ```

### 2. Service Account

Create a service account with Chronicle permissions:

```bash
# Create service account
gcloud iam service-accounts create chronicle-anomaly-detector \
    --display-name="Chronicle Anomaly Detector"

# Grant Chronicle permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:chronicle-anomaly-detector@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/chronicle.agent"

# Create and download key
gcloud iam service-accounts keys create chronicle_credentials.json \
    --iam-account=chronicle-anomaly-detector@PROJECT_ID.iam.gserviceaccount.com
```

**Alternative (UI):**
1. Go to IAM & Admin > Service Accounts
2. Click "Create Service Account"
3. Name: `chronicle-anomaly-detector`
4. Grant role: `Chronicle Agent`
5. Create key (JSON format)
6. Download and save as `chronicle_credentials.json`

### 3. Chronicle Customer ID

Find your Chronicle customer ID:

1. Go to [Chronicle console](https://chronicle.security.google.com/)
2. Settings > Customer Settings
3. Copy your Customer ID (format: `C00000000`)

### 4. Python Dependencies

```bash
pip install google-auth google-auth-httplib2 requests
```

Or use the complete requirements:
```bash
pip install -r requirements_chronicle.txt
```

---

## Setup Steps

### Step 1: Install Dependencies

```bash
cd C:\Users\jimmy\.local\bin

# Install Google Cloud and API dependencies
pip install google-auth google-auth-httplib2 requests fastapi uvicorn
```

### Step 2: Configure Chronicle

#### Option A: Interactive Setup
```bash
python google_chronicle_integration.py --setup
```

Follow the prompts:
```
Service account JSON path: chronicle_credentials.json
Chronicle customer ID: C00000000
Region (us/europe/asia): us
```

#### Option B: Manual Configuration

Create `chronicle_config.json`:
```json
{
  "credentials_file": "chronicle_credentials.json",
  "customer_id": "C00000000",
  "region": "us",
  "log_type": "SECURITY_ANOMALY",
  "batch_size": 100,
  "min_severity": "medium",
  "auto_create_rules": true,
  "detection_rules": [
    {
      "name": "High_Severity_Anomalies",
      "description": "Alert on high/critical severity anomalies",
      "min_severity": "HIGH",
      "min_score": 0.85
    }
  ]
}
```

### Step 3: Place Credentials File

Place your `chronicle_credentials.json` in the project directory:
```
C:\Users\jimmy\.local\bin\chronicle_credentials.json
```

**Security Note:** Add to `.gitignore`:
```
chronicle_credentials.json
chronicle_config.json
```

### Step 4: Test Connection

```bash
python google_chronicle_integration.py --test
```

Expected output:
```
Testing Chronicle connection...
✓ Successfully sent test event to Chronicle
```

---

## Configuration

### Configuration File (`chronicle_config.json`)

```json
{
  "credentials_file": "chronicle_credentials.json",
  "customer_id": "C00000000",
  "region": "us",
  "log_type": "SECURITY_ANOMALY",
  "batch_size": 100,
  "min_severity": "medium",
  "auto_create_rules": true,
  "detection_rules": [
    {
      "name": "Critical_Anomalies",
      "description": "Immediate alert on critical anomalies",
      "min_severity": "CRITICAL",
      "min_score": 0.95
    },
    {
      "name": "Brute_Force_Attacks",
      "description": "Detect credential brute force attempts",
      "min_severity": "MEDIUM",
      "min_score": 0.7,
      "threat_type": "brute_force"
    },
    {
      "name": "Privilege_Escalation",
      "description": "Detect privilege escalation attempts",
      "min_severity": "HIGH",
      "min_score": 0.8,
      "threat_type": "privilege_escalation"
    },
    {
      "name": "Data_Exfiltration",
      "description": "Detect potential data exfiltration",
      "min_severity": "HIGH",
      "min_score": 0.85,
      "threat_type": "data_exfiltration"
    }
  ]
}
```

### Regions

Chronicle is available in three regions:
- **us** - United States (default)
- **europe** - Europe
- **asia** - Asia (asia-southeast1)

---

## Usage

### Method 1: API with Chronicle Integration

```bash
# Start API with Chronicle integration
python anomaly_api_chronicle.py
```

The API automatically:
- Loads models on startup
- Enables Chronicle if configured
- Sends anomalies in background

**Test the API:**
```bash
python test_api.py
```

### Method 2: Direct Python Integration

```python
from google_chronicle_integration import ChronicleClient

# Initialize Chronicle client
chronicle = ChronicleClient(
    credentials_file="chronicle_credentials.json",
    customer_id="C00000000",
    region="us"
)

# Send anomalies
anomalies = [
    {
        "timestamp": "2026-01-15T10:00:00Z",
        "user": "admin",
        "source_ip": "10.0.0.100",
        "dest_ip": "unknown",
        "event_type": "login",
        "action": "failed",
        "message": "Failed login attempt",
        "severity": "high",
        "anomaly_score": 0.92,
        "threat_type": "brute_force"
    }
]

result = chronicle.send_anomalies(anomalies)
print(result)
```

### Method 3: Batch Processing with Chronicle

```python
from batch_processor import BatchProcessor
from google_chronicle_integration import ChronicleClient

# Initialize processors
processor = BatchProcessor(
    model_dir="anomaly_outputs",
    log_dir="logs",
    output_dir="batch_outputs"
)

chronicle = ChronicleClient(
    credentials_file="chronicle_credentials.json",
    customer_id="C00000000"
)

# Process and send to Chronicle
processor.load_models()
result = processor.process_batch()

# Send anomalies to Chronicle
for file_result in result['results']:
    if file_result['status'] == 'success':
        # Load anomalies from output file
        import json
        with open(file_result['output_file'], 'r') as f:
            data = json.load(f)
            chronicle.send_anomalies(data['anomalies'])
```

### Method 4: REST API Calls

```bash
# Enable Chronicle integration
curl -X POST http://localhost:8000/chronicle/enable \
  -H "Content-Type: application/json" \
  -d '{
    "credentials_file": "chronicle_credentials.json",
    "customer_id": "C00000000",
    "region": "us"
  }'

# Check Chronicle status
curl http://localhost:8000/chronicle/status

# Test Chronicle connection
curl -X POST http://localhost:8000/chronicle/test

# Analyze logs (automatically sends to Chronicle)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [...],
    "send_to_chronicle": true
  }'
```

---

## Detection Rules

### Create Detection Rules in Chronicle

The integration can automatically create YARA-L detection rules in Chronicle:

```bash
python google_chronicle_integration.py --create-rules
```

This creates rules based on `chronicle_config.json` configuration.

### Example YARA-L Rule

The integration generates rules like this:

```yara
rule High_Severity_Anomalies {
  meta:
    author = "Anomaly Detection System"
    description = "Alert on high/critical severity anomalies"
    severity = "HIGH"

  events:
    $anomaly.metadata.log_type = "SECURITY_ANOMALY"
    $anomaly.security_result.severity = /HIGH|CRITICAL/
    $anomaly.additional.fields.key = "anomaly_score"
    $anomaly.additional.fields.value.string_value >= "0.85"

  condition:
    $anomaly
}
```

### Viewing Alerts in Chronicle

1. Go to [Chronicle Console](https://chronicle.security.google.com/)
2. Navigate to **Detections** > **Rules**
3. Your rules will appear with the names from config
4. Alerts trigger when anomalies match rule conditions

---

## UDM Mapping

Anomalies are converted to Chronicle's Unified Data Model (UDM):

| Anomaly Field | UDM Field | Notes |
|--------------|-----------|-------|
| timestamp | metadata.event_timestamp | ISO 8601 format |
| user | principal.user.userid | Username |
| source_ip | principal.ip | Source IP address |
| dest_ip | target.ip | Destination IP (if applicable) |
| event_type | metadata.event_type | Mapped to UDM event types |
| severity | security_result.severity | LOW/MEDIUM/HIGH/CRITICAL |
| threat_type | security_result.category_details | Threat classification |
| anomaly_score | additional.fields | Custom field |
| message | metadata.description | Log message |

### UDM Event Type Mapping

| Threat Type | UDM Event Type |
|-------------|---------------|
| brute_force | USER_LOGIN |
| privilege_escalation | USER_UNCATEGORIZED |
| data_exfiltration | NETWORK_CONNECTION |
| lateral_movement | NETWORK_CONNECTION |
| unknown | GENERIC_EVENT |

---

## Monitoring & Verification

### Check Ingestion in Chronicle

1. Go to Chronicle Console
2. Navigate to **Search** > **Raw Logs**
3. Search for:
   ```
   metadata.log_type = "SECURITY_ANOMALY"
   ```
4. Verify your anomalies appear

### Query Examples

**Find all anomalies:**
```
metadata.log_type = "SECURITY_ANOMALY"
```

**High severity only:**
```
metadata.log_type = "SECURITY_ANOMALY" AND
security_result.severity = /HIGH|CRITICAL/
```

**Specific threat type:**
```
metadata.log_type = "SECURITY_ANOMALY" AND
security_result.category_details = "brute_force"
```

**By user:**
```
metadata.log_type = "SECURITY_ANOMALY" AND
principal.user.userid = "admin"
```

**By anomaly score:**
```
metadata.log_type = "SECURITY_ANOMALY" AND
additional.fields.key = "anomaly_score" AND
additional.fields.value.string_value >= "0.9"
```

---

## Troubleshooting

### Error: "Credentials file not found"

**Solution:**
```bash
# Verify file exists
ls chronicle_credentials.json

# Check path in config
cat chronicle_config.json | grep credentials_file
```

### Error: "Permission denied" or "401 Unauthorized"

**Solution:**
1. Verify service account has Chronicle Agent role
2. Check credentials file is valid
3. Regenerate service account key if needed

```bash
# Check service account permissions
gcloud projects get-iam-policy PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:chronicle-anomaly-detector*"
```

### Error: "Customer ID not found"

**Solution:**
1. Verify customer ID format (should be `C00000000`)
2. Check Chronicle console for correct ID
3. Ensure Chronicle is enabled for your project

### Events not appearing in Chronicle

**Possible causes:**

1. **Ingestion delay** - Wait 2-5 minutes for events to appear
2. **Wrong region** - Verify region matches your Chronicle instance
3. **Log type mismatch** - Ensure log_type is consistent

**Debug:**
```bash
# Enable debug logging
export CHRONICLE_DEBUG=1
python anomaly_api_chronicle.py
```

### High latency

**Optimization:**

1. **Use batching** - Set batch_size to 100-500
2. **Background sending** - API already uses background tasks
3. **Compress payloads** - Enable compression in requests

```python
# Optimize batch size
chronicle = ChronicleClient(...)
streamer = ChronicleStreamer(chronicle, batch_size=500)
```

---

## Production Deployment

### Docker with Chronicle

Update `docker-compose.yml`:

```yaml
services:
  anomaly-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./anomaly_outputs:/app/anomaly_outputs
      - ./chronicle_credentials.json:/app/chronicle_credentials.json:ro
      - ./chronicle_config.json:/app/chronicle_config.json:ro
    environment:
      - CHRONICLE_ENABLED=true
      - CHRONICLE_CUSTOMER_ID=C00000000
      - CHRONICLE_REGION=us
    command: python anomaly_api_chronicle.py
```

### Kubernetes Secret

Store credentials as secret:

```bash
kubectl create secret generic chronicle-creds \
  --from-file=chronicle_credentials.json
```

Reference in deployment:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: anomaly-detector
spec:
  containers:
  - name: api
    image: anomaly-detection:latest
    env:
    - name: CHRONICLE_CREDENTIALS
      value: /secrets/chronicle_credentials.json
    volumeMounts:
    - name: chronicle-secret
      mountPath: /secrets
      readOnly: true
  volumes:
  - name: chronicle-secret
    secret:
      secretName: chronicle-creds
```

### Rate Limiting

Chronicle API limits:
- **Ingestion**: 10,000 events/minute
- **Queries**: 1,000 requests/minute

The integration handles this with:
- Batching (configurable batch_size)
- Background processing
- Automatic retries

---

## Security Best Practices

1. **Protect credentials file**
   ```bash
   chmod 600 chronicle_credentials.json
   ```

2. **Use environment variables**
   ```bash
   export CHRONICLE_CREDENTIALS=/secure/path/credentials.json
   export CHRONICLE_CUSTOMER_ID=C00000000
   ```

3. **Rotate service account keys**
   - Rotate every 90 days
   - Use multiple keys for zero-downtime rotation

4. **Network security**
   - Use VPC Service Controls
   - Enable Private Google Access
   - Restrict API access by IP

5. **Audit logging**
   - Enable Cloud Audit Logs
   - Monitor Chronicle access logs
   - Alert on unusual API usage

---

## Support & Resources

- [Chronicle Documentation](https://cloud.google.com/chronicle/docs)
- [Chronicle API Reference](https://cloud.google.com/chronicle/docs/reference/rest)
- [UDM Field List](https://cloud.google.com/chronicle/docs/reference/udm-field-list)
- [YARA-L Rules](https://cloud.google.com/chronicle/docs/detection/yara-l-2-0-overview)

---

## Summary

You now have a complete Google Chronicle SIEM integration:

✅ Automatic anomaly forwarding
✅ Real-time and batch processing
✅ Custom detection rules
✅ UDM format conversion
✅ Production-ready deployment

**Next Steps:**
1. Configure Chronicle credentials
2. Test the connection
3. Create detection rules
4. Deploy to production
5. Monitor alerts in Chronicle console
