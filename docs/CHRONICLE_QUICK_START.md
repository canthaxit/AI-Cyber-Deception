# Google Chronicle SIEM - Quick Start

Get your anomaly detection system connected to Google Chronicle SIEM in 5 minutes.

## Prerequisites

- ✅ Google Cloud project with Chronicle enabled
- ✅ Service account with Chronicle Agent role
- ✅ Service account JSON key file

## Step 1: Install Dependencies (1 minute)

```bash
pip install google-auth google-auth-httplib2 requests fastapi uvicorn
```

Or use the complete requirements:
```bash
pip install -r requirements_chronicle.txt
```

## Step 2: Get Service Account Credentials (2 minutes)

### Option A: Using gcloud CLI

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create service account
gcloud iam service-accounts create chronicle-anomaly-detector \
    --display-name="Chronicle Anomaly Detector"

# Grant Chronicle Agent role
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:chronicle-anomaly-detector@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/chronicle.agent"

# Create key
gcloud iam service-accounts keys create chronicle_credentials.json \
    --iam-account=chronicle-anomaly-detector@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### Option B: Using Google Cloud Console

1. Go to [IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
2. Click **Create Service Account**
3. Name: `chronicle-anomaly-detector`
4. Click **Create and Continue**
5. Select role: **Chronicle Agent**
6. Click **Continue**, then **Done**
7. Click on the service account
8. Go to **Keys** tab
9. Click **Add Key** > **Create new key**
10. Select **JSON**, click **Create**
11. Save the downloaded file as `chronicle_credentials.json`

## Step 3: Configure Chronicle (1 minute)

### Option A: Automated Setup

```bash
# Windows
setup_chronicle.bat

# Linux/Mac
chmod +x setup_chronicle.sh
./setup_chronicle.sh
```

Follow the prompts to enter:
- Customer ID (from Chronicle console)
- Region (us/europe/asia)

### Option B: Manual Configuration

1. Copy the template:
   ```bash
   copy chronicle_config_template.json chronicle_config.json
   ```

2. Edit `chronicle_config.json`:
   ```json
   {
     "credentials_file": "chronicle_credentials.json",
     "customer_id": "C00000000",
     "region": "us",
     "log_type": "SECURITY_ANOMALY"
   }
   ```

3. Get your Customer ID:
   - Go to [Chronicle Console](https://chronicle.security.google.com/)
   - Settings > Customer Settings
   - Copy Customer ID (format: `C00000000`)

## Step 4: Test Connection (1 minute)

```bash
python google_chronicle_integration.py --test
```

Expected output:
```
Testing Chronicle connection...
✓ Successfully sent test event to Chronicle
```

## Step 5: Start Using It!

### Method 1: API with Auto-Forward

```bash
# Start API with Chronicle integration
python anomaly_api_chronicle.py

# API auto-loads models and enables Chronicle if configured
# All detected anomalies are automatically sent to Chronicle
```

Test it:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [{
      "timestamp": "2026-01-15T10:00:00Z",
      "user": "admin",
      "source_ip": "10.0.0.100",
      "event_type": "login",
      "action": "failed",
      "message": "Failed login"
    }],
    "send_to_chronicle": true
  }'
```

### Method 2: Direct Python Integration

```python
from google_chronicle_integration import ChronicleClient

# Initialize
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
        "event_type": "login",
        "action": "failed",
        "message": "Failed login",
        "severity": "high",
        "anomaly_score": 0.92,
        "threat_type": "brute_force"
    }
]

result = chronicle.send_anomalies(anomalies)
print(result)
```

### Method 3: Batch Processing

```python
from batch_processor import BatchProcessor
from google_chronicle_integration import ChronicleClient

# Initialize
processor = BatchProcessor(log_dir="logs", output_dir="batch_outputs")
chronicle = ChronicleClient(
    credentials_file="chronicle_credentials.json",
    customer_id="C00000000"
)

# Process and send
processor.load_models()
result = processor.process_batch()

# Send to Chronicle
for file_result in result['results']:
    import json
    with open(file_result['output_file']) as f:
        data = json.load(f)
        chronicle.send_anomalies(data['anomalies'])
```

## Verify in Chronicle Console

1. Go to [Chronicle Console](https://chronicle.security.google.com/)
2. Navigate to **Search** > **Raw Logs**
3. Search for:
   ```
   metadata.log_type = "SECURITY_ANOMALY"
   ```
4. You should see your anomalies!

## Create Detection Rules (Optional)

```bash
python google_chronicle_integration.py --create-rules
```

This creates YARA-L detection rules in Chronicle based on your config.

View them in Chronicle:
- **Detections** > **Rules**

## Common Queries in Chronicle

**All anomalies:**
```
metadata.log_type = "SECURITY_ANOMALY"
```

**High severity only:**
```
metadata.log_type = "SECURITY_ANOMALY" AND
security_result.severity = /HIGH|CRITICAL/
```

**Brute force attacks:**
```
metadata.log_type = "SECURITY_ANOMALY" AND
security_result.category_details = "brute_force"
```

**Specific user:**
```
metadata.log_type = "SECURITY_ANOMALY" AND
principal.user.userid = "admin"
```

**High anomaly scores:**
```
metadata.log_type = "SECURITY_ANOMALY" AND
additional.fields.key = "anomaly_score" AND
additional.fields.value.string_value >= "0.9"
```

## Troubleshooting

### "Credentials file not found"
```bash
# Check if file exists
ls chronicle_credentials.json

# Make sure it's in the current directory
pwd
```

### "Permission denied"
```bash
# Verify service account has Chronicle Agent role
gcloud projects get-iam-policy YOUR_PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:chronicle-anomaly-detector*"
```

### Events not showing up
- Wait 2-5 minutes (ingestion delay)
- Check region matches your Chronicle instance
- Verify customer ID is correct

## Next Steps

1. **Enable auto-forwarding** - All anomalies automatically sent to Chronicle
2. **Create detection rules** - Alert on specific threat patterns
3. **Set up dashboards** - Visualize threats in Chronicle
4. **Configure alerts** - Get notified of critical events

## Resources

- **Detailed Guide**: [GOOGLE_SIEM_SETUP.md](GOOGLE_SIEM_SETUP.md)
- **Integration Examples**: [integration_examples.py](integration_examples.py)
- **Chronicle Docs**: https://cloud.google.com/chronicle/docs
- **API Reference**: https://cloud.google.com/chronicle/docs/reference/rest

---

**You're now connected to Google Chronicle SIEM!** 🎉

All detected anomalies will appear in your Chronicle console with full context, severity, and threat classification.
