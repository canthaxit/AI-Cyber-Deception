# Integration Examples

Code examples for integrating with various platforms and services.

## Files

- **integration_examples.py** - Complete integration examples for 10+ platforms

## Available Examples

### 1. Direct Python Integration
```python
from integration_examples import example_direct_integration

anomalies = example_direct_integration()
```

### 2. REST API Client
```python
from integration_examples import AnomalyDetectionClient

client = AnomalyDetectionClient("http://localhost:8000")
client.load_models()

result = client.analyze_logs([{
    "timestamp": "2026-01-15T10:00:00Z",
    "user": "admin",
    "source_ip": "10.0.0.100",
    "event_type": "login",
    "action": "failed",
    "message": "Failed login"
}])

print(f"Detected {result['anomalies_detected']} anomalies")
```

### 3. Google Chronicle SIEM
```python
from integration_examples import ChronicleIntegration

chronicle = ChronicleIntegration(
    credentials_file="../chronicle/chronicle_credentials.json",
    customer_id="C00000000",
    region="us"
)

chronicle.send_anomalies(anomalies)
```

### 4. Slack Alerts
```python
from integration_examples import SlackAlerter

slack = SlackAlerter("https://hooks.slack.com/services/YOUR/WEBHOOK/URL")

for anomaly in anomalies:
    if anomaly['severity'] in ['high', 'critical']:
        slack.send_alert(anomaly)
```

### 5. Splunk HEC
```python
from integration_examples import SplunkIntegration

splunk = SplunkIntegration(
    hec_url="https://splunk.example.com:8088/services/collector",
    token="YOUR-HEC-TOKEN"
)

for anomaly in anomalies:
    splunk.send_event(anomaly)
```

### 6. Elasticsearch
```python
from integration_examples import ElasticsearchIntegration

es = ElasticsearchIntegration("http://localhost:9200")
es.bulk_index(anomalies)
```

### 7. Webhook Alerts
```python
from integration_examples import WebhookAlerter

webhook = WebhookAlerter("https://your-webhook.com/alerts")
webhook.send_alert(anomaly)
```

### 8. Generic SIEM (Syslog)
```python
from integration_examples import SIEMIntegration

siem = SIEMIntegration(syslog_host="siem.example.com", syslog_port=514)
siem.send_syslog(anomaly)
```

### 9. Kafka Streaming
```python
from integration_examples import example_kafka_integration

example_kafka_integration()
```

### 10. AWS Lambda
```python
from integration_examples import lambda_handler

# Deploy as AWS Lambda function
# Handles log analysis in serverless environment
```

## Running Examples

### Start the API
```bash
cd ../api/
python anomaly_api.py
```

### Run Examples
```python
cd examples/
python integration_examples.py
```

This will:
1. Test REST API connection
2. Show MCP configuration
3. Display all available integrations
4. Provide setup instructions

## Custom Integration Template

```python
class CustomIntegration:
    """Template for custom integration."""

    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url

    def analyze_and_forward(self, logs):
        """Analyze logs and forward to custom system."""
        import requests

        # Analyze with API
        response = requests.post(
            f"{self.api_url}/analyze",
            json={"logs": logs}
        )

        anomalies = response.json()['anomalies']

        # Forward to custom system
        for anomaly in anomalies:
            self.send_to_custom_system(anomaly)

        return anomalies

    def send_to_custom_system(self, anomaly):
        """Send to your custom system."""
        # Implement your custom logic here
        pass
```

## Platform-Specific Guides

### Cloud Platforms

**AWS:**
- Lambda: Serverless function deployment
- Kinesis: Stream processing
- S3: Log storage integration
- CloudWatch: Monitoring integration

**Google Cloud:**
- Cloud Run: Container deployment
- Chronicle: SIEM integration
- Cloud Functions: Serverless
- BigQuery: Data warehouse integration

**Azure:**
- Container Instances: Container deployment
- Functions: Serverless
- Sentinel: SIEM integration
- Log Analytics: Monitoring

### SIEM Platforms

**Google Chronicle:**
```python
from integration_examples import example_chronicle_integration
example_chronicle_integration()
```

**Splunk:**
```python
from integration_examples import SplunkIntegration
splunk = SplunkIntegration(hec_url, token)
splunk.send_event(anomaly)
```

**Elasticsearch/OpenSearch:**
```python
from integration_examples import ElasticsearchIntegration
es = ElasticsearchIntegration(url)
es.bulk_index(anomalies)
```

### Messaging & Alerts

**Slack:**
```python
slack = SlackAlerter(webhook_url)
slack.send_alert(anomaly)
```

**Email (SMTP):**
```python
import smtplib
from email.mime.text import MIMEText

msg = MIMEText(json.dumps(anomaly, indent=2))
msg['Subject'] = f"Security Alert: {anomaly['threat_type']}"
smtp = smtplib.SMTP('localhost')
smtp.send_message(msg)
```

**PagerDuty:**
```python
import requests

requests.post(
    "https://events.pagerduty.com/v2/enqueue",
    json={
        "routing_key": "YOUR_INTEGRATION_KEY",
        "event_action": "trigger",
        "payload": {
            "summary": f"Security Alert: {anomaly['threat_type']}",
            "severity": anomaly['severity'],
            "source": anomaly['source_ip']
        }
    }
)
```

### Streaming Platforms

**Kafka:**
```python
from kafka import KafkaProducer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
producer.send('security-anomalies', json.dumps(anomaly).encode())
```

**AWS Kinesis:**
```python
import boto3
kinesis = boto3.client('kinesis')
kinesis.put_record(
    StreamName='security-anomalies',
    Data=json.dumps(anomaly),
    PartitionKey=anomaly['user']
)
```

**Azure Event Hubs:**
```python
from azure.eventhub import EventHubProducerClient
producer = EventHubProducerClient.from_connection_string(conn_str)
producer.send_event(json.dumps(anomaly))
```

## Best Practices

### Error Handling
```python
import logging

try:
    result = integration.send_anomaly(anomaly)
except Exception as e:
    logging.error(f"Failed to send anomaly: {e}")
    # Implement retry logic or fallback
```

### Batching
```python
# Batch for better performance
batch = []
for anomaly in anomalies:
    batch.append(anomaly)
    if len(batch) >= 100:
        integration.send_batch(batch)
        batch = []

# Send remaining
if batch:
    integration.send_batch(batch)
```

### Async Processing
```python
import asyncio

async def send_async(anomaly):
    # Implement async sending
    pass

# Send in parallel
await asyncio.gather(*[send_async(a) for a in anomalies])
```

### Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def send_with_retry(anomaly):
    return integration.send_anomaly(anomaly)
```

## Testing Integrations

```python
# Test with sample anomaly
test_anomaly = {
    "timestamp": "2026-01-15T10:00:00Z",
    "user": "test_user",
    "source_ip": "192.168.1.100",
    "event_type": "login",
    "action": "failed",
    "message": "Test event",
    "severity": "low",
    "anomaly_score": 0.65,
    "threat_type": "brute_force"
}

# Test each integration
slack.send_alert(test_anomaly)
splunk.send_event(test_anomaly)
es.index_anomaly(test_anomaly)
```

## Documentation

- **Integration Guide**: [`../docs/README_INTEGRATIONS.md`](../docs/README_INTEGRATIONS.md)
- **API Reference**: [`../api/README.md`](../api/README.md)
- **Chronicle Setup**: [`../chronicle/README.md`](../chronicle/README.md)

## Contributing

To add a new integration example:

1. Add class/function to `integration_examples.py`
2. Follow existing naming convention
3. Include docstrings with usage examples
4. Add to main demo section
5. Update this README

## Support

For integration-specific issues:
- Check platform documentation
- Verify credentials and permissions
- Test with sample data first
- Enable debug logging
