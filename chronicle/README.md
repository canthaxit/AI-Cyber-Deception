# Google Chronicle SIEM Integration

Integration with Google Security Operations (Chronicle) SIEM platform.

## Files

- **google_chronicle_integration.py** - Chronicle client library
- **setup_chronicle.bat** - Automated setup wizard
- **chronicle_config_template.json** - Configuration template

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r ../config/requirements_chronicle.txt
```

### 2. Run Setup
```bash
setup_chronicle.bat
```

You'll need:
- **Service account JSON** from Google Cloud Console
- **Chronicle Customer ID** from Chronicle console (format: `C00000000`)
- **Region**: us, europe, or asia

### 3. Test Connection
```bash
python google_chronicle_integration.py --test
```

Expected output:
```
✓ Successfully sent test event to Chronicle
```

## Usage

### Direct Python Integration

```python
from google_chronicle_integration import ChronicleClient

# Initialize client
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

### With API Integration

Use the Chronicle-enabled API:
```bash
cd ../api/
python anomaly_api_chronicle.py
```

All detected anomalies are automatically sent to Chronicle!

### CLI Commands

```bash
# Setup configuration
python google_chronicle_integration.py --setup

# Test connection
python google_chronicle_integration.py --test

# Send anomalies from file
python google_chronicle_integration.py --send-anomalies ../tests/anomalies.json

# Create detection rules
python google_chronicle_integration.py --create-rules
```

## Configuration

### chronicle_config.json

```json
{
  "credentials_file": "chronicle_credentials.json",
  "customer_id": "C00000000",
  "region": "us",
  "log_type": "SECURITY_ANOMALY",
  "batch_size": 100,
  "detection_rules": [
    {
      "name": "High_Severity_Anomalies",
      "description": "Alert on high/critical anomalies",
      "min_severity": "HIGH",
      "min_score": 0.85
    }
  ]
}
```

### Get Service Account Credentials

**Option A: Using gcloud CLI**
```bash
# Create service account
gcloud iam service-accounts create chronicle-anomaly-detector

# Grant Chronicle Agent role
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:chronicle-anomaly-detector@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/chronicle.agent"

# Create key
gcloud iam service-accounts keys create chronicle_credentials.json \
  --iam-account=chronicle-anomaly-detector@PROJECT_ID.iam.gserviceaccount.com
```

**Option B: Google Cloud Console**
1. IAM & Admin > Service Accounts
2. Create Service Account: `chronicle-anomaly-detector`
3. Grant role: Chronicle Agent
4. Create JSON key
5. Save as `chronicle_credentials.json`

## View Results in Chronicle

1. Go to [Chronicle Console](https://chronicle.security.google.com/)
2. Navigate to **Search** > **Raw Logs**
3. Search for:
   ```
   metadata.log_type = "SECURITY_ANOMALY"
   ```

### Useful Queries

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

**By user:**
```
metadata.log_type = "SECURITY_ANOMALY" AND
principal.user.userid = "admin"
```

## Detection Rules

Create YARA-L detection rules:
```bash
python google_chronicle_integration.py --create-rules
```

View in Chronicle:
- **Detections** > **Rules**

## UDM Mapping

Anomalies are converted to Chronicle's Unified Data Model:

| Anomaly Field | Chronicle UDM Field |
|--------------|---------------------|
| timestamp | metadata.event_timestamp |
| user | principal.user.userid |
| source_ip | principal.ip |
| dest_ip | target.ip |
| event_type | metadata.event_type |
| severity | security_result.severity |
| threat_type | security_result.category_details |
| anomaly_score | additional.fields |

## Documentation

- **Quick Start**: [`../docs/CHRONICLE_QUICK_START.md`](../docs/CHRONICLE_QUICK_START.md)
- **Detailed Guide**: [`../docs/GOOGLE_SIEM_SETUP.md`](../docs/GOOGLE_SIEM_SETUP.md)
- **Integration Options**: [`../docs/README_INTEGRATIONS.md`](../docs/README_INTEGRATIONS.md)

## Troubleshooting

**"Credentials file not found"**
- Ensure `chronicle_credentials.json` is in this directory
- Check the path in `chronicle_config.json`

**"Permission denied"**
- Verify service account has Chronicle Agent role
- Regenerate credentials if needed

**Events not appearing**
- Wait 2-5 minutes for ingestion delay
- Check region matches your Chronicle instance
- Verify customer ID is correct

## Security

- Keep `chronicle_credentials.json` secure
- Add to `.gitignore`
- Rotate keys every 90 days
- Use environment variables for production
