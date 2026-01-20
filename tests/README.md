# Test Data

Sample log files for testing anomaly detection.

## Files

- **test_logs_normal.json** - Normal behavior baseline (10 events)
- **test_logs_attack.json** - Attack scenarios (18 events)
- **test_logs/** - Additional test data (if created)

## Test Data Overview

### test_logs_normal.json
Contains normal user behavior:
- alice: login, file access, logout
- bob: login, logout
- charlie: login, logout

All during business hours, successful actions.

### test_logs_attack.json
Contains attack scenarios:
1. **Brute force attack** (12 events)
   - User: admin
   - IP: 10.0.0.100
   - Pattern: 12 consecutive failed logins in 35 seconds

2. **Data exfiltration** (1 event)
   - User: root
   - Time: 2:00 AM (off-hours)
   - Action: Accessed /etc/shadow

## Running Tests

### Train Models
```bash
cd ../core/
python log_anomaly_detection_lite.py \
  --data_path ../tests/test_logs_*.json \
  --baseline_period_days 1 \
  --contamination 0.50
```

Expected results:
- **Total events**: 28 (10 normal + 18 attack)
- **Anomalies detected**: ~9 (50%)
- **Threat types**: brute_force (8), data_exfiltration (1)

### Test API
```bash
cd ../api/
python anomaly_api.py &
python test_api.py
```

### Test Batch Processing
```bash
cd ../batch/
python batch_processor.py \
  --log-dir ../tests/ \
  --output-dir ./batch_outputs/ \
  --once
```

## Creating Custom Test Data

### Normal Behavior Template
```json
{
  "timestamp": "2026-01-15T09:00:00Z",
  "user": "alice",
  "source_ip": "192.168.1.10",
  "dest_ip": "unknown",
  "event_type": "login",
  "action": "success",
  "message": "User logged in"
}
```

### Attack Pattern Templates

**Brute Force:**
```json
[
  {
    "timestamp": "2026-01-15T10:00:00Z",
    "user": "admin",
    "source_ip": "10.0.0.100",
    "event_type": "login",
    "action": "failed",
    "message": "Failed login attempt"
  },
  {
    "timestamp": "2026-01-15T10:00:05Z",
    "user": "admin",
    "source_ip": "10.0.0.100",
    "event_type": "login",
    "action": "failed",
    "message": "Failed login attempt"
  }
  // ... repeat 10+ times
]
```

**Privilege Escalation:**
```json
{
  "timestamp": "2026-01-15T14:30:00Z",
  "user": "lowpriv",
  "source_ip": "192.168.1.50",
  "event_type": "sudo",
  "action": "success",
  "message": "sudo command executed: /bin/bash"
}
```

**Data Exfiltration:**
```json
{
  "timestamp": "2026-01-15T02:00:00Z",
  "user": "dbuser",
  "source_ip": "192.168.1.20",
  "event_type": "file_access",
  "action": "success",
  "message": "Large file download: /var/db/customer_data.sql (500MB)"
}
```

**Lateral Movement:**
```json
{
  "timestamp": "2026-01-15T15:00:00Z",
  "user": "admin",
  "source_ip": "10.0.0.25",
  "event_type": "network",
  "action": "success",
  "message": "SSH connection to 192.168.1.100"
}
```

## Validation

### Check Detection Accuracy

```python
import json
import pandas as pd

# Load test results
with open('../core/anomaly_outputs/anomalies_detailed.json') as f:
    detected = json.load(f)

# Verify brute force detection
brute_force = [a for a in detected if a['threat_type'] == 'brute_force']
print(f"Brute force attacks detected: {len(brute_force)}")

# Verify data exfiltration detection
data_exfil = [a for a in detected if a['threat_type'] == 'data_exfiltration']
print(f"Data exfiltration detected: {len(data_exfil)}")

# Check false positives (should be low)
normal_flagged = [a for a in detected if a['user'] in ['alice', 'bob', 'charlie']]
print(f"False positives: {len(normal_flagged)}")
```

### Performance Metrics

```python
import time
from log_anomaly_detection_lite import LogParser, LogFeaturePipeline
import joblib

# Load models
pipeline = joblib.load('../core/anomaly_outputs/feature_pipeline.pkl')
iso_forest = joblib.load('../core/anomaly_outputs/isolation_forest_model.pkl')

# Load test data
parser = LogParser()
df = parser.load_logs('test_logs_attack.json')

# Time the analysis
start = time.time()
features = pipeline.transform(df)
scores = iso_forest.score_samples(features)
duration = time.time() - start

print(f"Analyzed {len(df)} events in {duration:.3f}s")
print(f"Throughput: {len(df)/duration:.0f} events/second")
```

## Expected Results

### Detection Rates
- **Brute force**: 90-100% detection
- **Data exfiltration**: 80-90% detection
- **Privilege escalation**: 70-80% detection
- **False positive rate**: < 5%

### Performance
- **Throughput**: 500-1000 events/second
- **Latency**: 20-50ms per event
- **Memory**: ~500MB for trained models

## Adding New Test Cases

1. Create JSON file with test events
2. Place in `tests/` directory
3. Run detection:
   ```bash
   cd ../core/
   python log_anomaly_detection_lite.py --data_path ../tests/your_test.json
   ```
4. Verify results in `anomaly_outputs/`

## Integration Testing

### Full Pipeline Test
```bash
# 1. Train models
cd ../core/ && python log_anomaly_detection_lite.py --data_path ../tests/

# 2. Test API
cd ../api/ && python anomaly_api.py &
cd ../api/ && python test_api.py

# 3. Test batch processing
cd ../batch/ && python batch_processor.py --log-dir ../tests/ --once

# 4. Test MCP (requires Claude Desktop)
cd ../mcp/ && python anomaly_mcp_server.py
```

## Documentation

- **Quick Start**: [`../docs/QUICK_START.md`](../docs/QUICK_START.md)
- **Testing Guide**: [`../docs/LITE_VERSION_GUIDE.md`](../docs/LITE_VERSION_GUIDE.md)
