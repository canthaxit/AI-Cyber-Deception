#!/usr/bin/env python3
"""
Integration Examples for Log Anomaly Detection
Shows how to integrate with various platforms and AI tools
"""

import json
import requests
from typing import List, Dict, Any


# ============================================================================
# Example 1: Direct Python Integration
# ============================================================================

def example_direct_integration():
    """Use the detection pipeline directly in Python code."""
    from log_anomaly_detection_lite import (
        LogParser, LogFeaturePipeline, StatisticalAnomalyDetector,
        AnomalyScorer, preprocess_logs
    )
    from sklearn.ensemble import IsolationForest
    import joblib

    # Load models
    feature_pipeline = joblib.load("anomaly_outputs/feature_pipeline.pkl")
    iso_forest = joblib.load("anomaly_outputs/isolation_forest_model.pkl")
    stat_detector = joblib.load("anomaly_outputs/statistical_detector.pkl")
    scorer = AnomalyScorer()

    # Parse logs
    parser = LogParser()
    df = parser.load_logs("logs/")

    # Detect anomalies
    features = feature_pipeline.transform(df)
    iso_scores = -iso_forest.score_samples(features)
    stat_scores = stat_detector.detect_all(df)

    combined_scores = scorer.combine_scores({
        'isolation_forest': iso_scores,
        'statistical': stat_scores
    })

    # Get anomalies
    threshold = 0.7
    anomalies = df[combined_scores > threshold]

    print(f"Found {len(anomalies)} anomalies")
    return anomalies


# ============================================================================
# Example 2: REST API Integration
# ============================================================================

class AnomalyDetectionClient:
    """Client for REST API integration."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def load_models(self, model_dir: str = "anomaly_outputs") -> Dict[str, Any]:
        """Load detection models."""
        response = requests.post(
            f"{self.base_url}/models/load",
            params={"model_dir": model_dir}
        )
        return response.json()

    def analyze_logs(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze log events."""
        response = requests.post(
            f"{self.base_url}/analyze",
            json={"logs": logs}
        )
        return response.json()

    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """Analyze log file."""
        with open(filepath, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/analyze/file",
                files={"file": f}
            )
        return response.json()


def example_rest_api():
    """Example REST API usage."""
    client = AnomalyDetectionClient()

    # Check health
    health = client.health_check()
    print(f"API Status: {health['status']}")

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
            "message": "Failed login attempt"
        }
    ]

    result = client.analyze_logs(logs)
    print(f"Anomalies detected: {result['anomalies_detected']}")

    return result


# ============================================================================
# Example 3: MCP Integration (for Claude and other AI tools)
# ============================================================================

def example_mcp_config():
    """
    Example MCP configuration for Claude Desktop.

    Add this to your claude_desktop_config.json:
    """
    config = {
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

    print("Add this to your Claude Desktop config:")
    print(json.dumps(config, indent=2))

    return config


# ============================================================================
# Example 4: Webhook Integration (for real-time alerts)
# ============================================================================

class WebhookAlerter:
    """Send alerts to webhook endpoints."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_alert(self, anomaly: Dict[str, Any]):
        """Send anomaly alert to webhook."""
        alert = {
            "title": f"Security Alert: {anomaly['threat_type']}",
            "severity": anomaly['severity'],
            "user": anomaly['user'],
            "source_ip": anomaly['source_ip'],
            "message": anomaly['message'],
            "anomaly_score": anomaly['anomaly_score'],
            "timestamp": anomaly['timestamp']
        }

        response = requests.post(self.webhook_url, json=alert)
        return response.status_code == 200


def example_real_time_monitoring():
    """Monitor logs in real-time and send alerts."""
    client = AnomalyDetectionClient()
    alerter = WebhookAlerter("https://your-webhook-url.com/alerts")

    # Load models
    client.load_models()

    # Analyze logs (could be from a log stream)
    logs = get_recent_logs()  # Your log collection function

    result = client.analyze_logs(logs)

    # Send alerts for high-severity anomalies
    for anomaly in result['anomalies']:
        if anomaly['severity'] in ['high', 'critical']:
            alerter.send_alert(anomaly)
            print(f"Alert sent for {anomaly['threat_type']}")


def get_recent_logs():
    """Placeholder for log collection."""
    return []


# ============================================================================
# Example 5: Slack Integration
# ============================================================================

class SlackAlerter:
    """Send alerts to Slack."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_alert(self, anomaly: Dict[str, Any]):
        """Send formatted alert to Slack."""
        color = {
            'low': '#36a64f',
            'medium': '#ff9900',
            'high': '#ff0000',
            'critical': '#8b0000'
        }.get(anomaly['severity'], '#cccccc')

        message = {
            "attachments": [{
                "color": color,
                "title": f"🚨 Security Alert: {anomaly['threat_type']}",
                "fields": [
                    {"title": "Severity", "value": anomaly['severity'].upper(), "short": True},
                    {"title": "Score", "value": f"{anomaly['anomaly_score']:.3f}", "short": True},
                    {"title": "User", "value": anomaly['user'], "short": True},
                    {"title": "Source IP", "value": anomaly['source_ip'], "short": True},
                    {"title": "Event", "value": anomaly['event_type'], "short": True},
                    {"title": "Action", "value": anomaly['action'], "short": True},
                    {"title": "Message", "value": anomaly['message'], "short": False}
                ],
                "footer": "Log Anomaly Detection",
                "ts": anomaly['timestamp']
            }]
        }

        response = requests.post(self.webhook_url, json=message)
        return response.status_code == 200


# ============================================================================
# Example 6: Splunk Integration
# ============================================================================

class SplunkIntegration:
    """Send results to Splunk HEC (HTTP Event Collector)."""

    def __init__(self, hec_url: str, token: str):
        self.hec_url = hec_url
        self.headers = {
            "Authorization": f"Splunk {token}",
            "Content-Type": "application/json"
        }

    def send_event(self, anomaly: Dict[str, Any]):
        """Send anomaly to Splunk."""
        event = {
            "sourcetype": "anomaly_detection",
            "event": {
                "threat_type": anomaly['threat_type'],
                "severity": anomaly['severity'],
                "anomaly_score": anomaly['anomaly_score'],
                "user": anomaly['user'],
                "source_ip": anomaly['source_ip'],
                "event_type": anomaly['event_type'],
                "action": anomaly['action'],
                "message": anomaly['message'],
                "timestamp": anomaly['timestamp']
            }
        }

        response = requests.post(self.hec_url, headers=self.headers, json=event)
        return response.status_code == 200


# ============================================================================
# Example 7: Elasticsearch Integration
# ============================================================================

class ElasticsearchIntegration:
    """Index anomalies in Elasticsearch."""

    def __init__(self, es_url: str, index: str = "security-anomalies"):
        self.es_url = es_url
        self.index = index

    def index_anomaly(self, anomaly: Dict[str, Any]):
        """Index anomaly document."""
        response = requests.post(
            f"{self.es_url}/{self.index}/_doc",
            json=anomaly
        )
        return response.status_code in [200, 201]

    def bulk_index(self, anomalies: List[Dict[str, Any]]):
        """Bulk index multiple anomalies."""
        bulk_data = []
        for anomaly in anomalies:
            bulk_data.append({"index": {"_index": self.index}})
            bulk_data.append(anomaly)

        body = "\n".join(json.dumps(line) for line in bulk_data) + "\n"

        response = requests.post(
            f"{self.es_url}/_bulk",
            headers={"Content-Type": "application/x-ndjson"},
            data=body
        )
        return response.status_code == 200


# ============================================================================
# Example 8: SIEM Integration (Generic)
# ============================================================================

class SIEMIntegration:
    """Generic SIEM integration via syslog."""

    def __init__(self, syslog_host: str, syslog_port: int = 514):
        import socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.host = syslog_host
        self.port = syslog_port

    def send_syslog(self, anomaly: Dict[str, Any]):
        """Send anomaly as syslog message."""
        severity_map = {'low': 5, 'medium': 4, 'high': 2, 'critical': 1}
        severity = severity_map.get(anomaly['severity'], 5)

        message = (
            f"<{severity}>ANOMALY: {anomaly['threat_type']} | "
            f"User={anomaly['user']} IP={anomaly['source_ip']} "
            f"Score={anomaly['anomaly_score']:.3f} | {anomaly['message']}"
        )

        self.sock.sendto(message.encode(), (self.host, self.port))


# ============================================================================
# Example 9: Streaming Integration (Kafka)
# ============================================================================

def example_kafka_integration():
    """
    Example Kafka producer integration.
    Requires: pip install kafka-python
    """
    try:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        client = AnomalyDetectionClient()
        client.load_models()

        # Analyze logs
        logs = get_recent_logs()
        result = client.analyze_logs(logs)

        # Send anomalies to Kafka topic
        for anomaly in result['anomalies']:
            producer.send('security-anomalies', anomaly)

        producer.flush()
        print(f"Sent {len(result['anomalies'])} anomalies to Kafka")

    except ImportError:
        print("kafka-python not installed")


# ============================================================================
# Example 10: Google Chronicle SIEM Integration
# ============================================================================

class ChronicleIntegration:
    """Integration with Google Chronicle SIEM."""

    def __init__(self, credentials_file: str, customer_id: str, region: str = "us"):
        from google_chronicle_integration import ChronicleClient

        self.chronicle = ChronicleClient(
            credentials_file=credentials_file,
            customer_id=customer_id,
            region=region
        )

    def send_anomalies(self, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send anomalies to Chronicle."""
        return self.chronicle.send_anomalies(anomalies)

    def create_detection_rules(self, rules: List[Dict[str, Any]]):
        """Create detection rules in Chronicle."""
        results = []
        for rule in rules:
            result = self.chronicle.create_detection_rule(rule)
            results.append(result)
        return results


def example_chronicle_integration():
    """
    Example Chronicle SIEM integration.
    Requires: pip install google-auth google-auth-httplib2
    """
    try:
        from google_chronicle_integration import ChronicleClient

        # Initialize Chronicle
        chronicle = ChronicleClient(
            credentials_file="chronicle_credentials.json",
            customer_id="C00000000",
            region="us"
        )

        # Analyze logs
        client = AnomalyDetectionClient()
        client.load_models()

        logs = get_recent_logs()
        result = client.analyze_logs(logs)

        # Send anomalies to Chronicle
        if result['anomalies']:
            chronicle_result = chronicle.send_anomalies(result['anomalies'])
            print(f"Sent {len(result['anomalies'])} anomalies to Chronicle")
            print(f"Result: {chronicle_result}")

        # Create detection rules
        rules = [
            {
                "name": "High_Severity_Anomalies",
                "description": "Alert on high severity security events",
                "min_severity": "HIGH",
                "min_score": 0.85
            }
        ]

        for rule in rules:
            rule_result = chronicle.create_detection_rule(rule)
            print(f"Created rule '{rule['name']}': {rule_result['status']}")

    except ImportError:
        print("Chronicle integration not installed")
        print("Install with: pip install google-auth google-auth-httplib2")


def example_chronicle_with_api():
    """Example using API with Chronicle integration."""
    import requests

    base_url = "http://localhost:8000"

    # Enable Chronicle integration
    response = requests.post(
        f"{base_url}/chronicle/enable",
        json={
            "credentials_file": "chronicle_credentials.json",
            "customer_id": "C00000000",
            "region": "us"
        }
    )
    print(f"Chronicle enabled: {response.json()}")

    # Check status
    status = requests.get(f"{base_url}/chronicle/status").json()
    print(f"Chronicle status: {status}")

    # Analyze logs (automatically sent to Chronicle)
    logs = [
        {
            "timestamp": "2026-01-15T10:00:00Z",
            "user": "admin",
            "source_ip": "10.0.0.100",
            "event_type": "login",
            "action": "failed",
            "message": "Failed login attempt"
        }
    ]

    response = requests.post(
        f"{base_url}/analyze",
        json={
            "logs": logs,
            "send_to_chronicle": True
        }
    )

    result = response.json()
    print(f"Analysis complete. Chronicle sent: {result.get('chronicle_sent')}")


# ============================================================================
# Example 11: AWS Lambda Integration
# ============================================================================

def lambda_handler(event, context):
    """
    AWS Lambda handler for serverless deployment.

    Deploy with:
    - Docker image containing the API
    - S3 bucket for models
    - CloudWatch for logs
    """
    import boto3

    # Download models from S3
    s3 = boto3.client('s3')
    s3.download_file('my-models-bucket', 'feature_pipeline.pkl', '/tmp/feature_pipeline.pkl')

    # Analyze logs from event
    logs = json.loads(event['body'])['logs']

    client = AnomalyDetectionClient(base_url=os.environ['API_URL'])
    result = client.analyze_logs(logs)

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }


# ============================================================================
# Main Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Log Anomaly Detection - Integration Examples")
    print("=" * 70)

    print("\n1. REST API Example:")
    print("-" * 70)
    try:
        result = example_rest_api()
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the API is running: python anomaly_api.py")

    print("\n2. MCP Configuration:")
    print("-" * 70)
    example_mcp_config()

    print("\n3. Available Integrations:")
    print("-" * 70)
    print("✓ Direct Python import")
    print("✓ REST API (FastAPI)")
    print("✓ MCP Server (Claude, AI platforms)")
    print("✓ Google Chronicle SIEM")
    print("✓ Webhooks (real-time alerts)")
    print("✓ Slack notifications")
    print("✓ Splunk HEC")
    print("✓ Elasticsearch indexing")
    print("✓ Generic SIEM (syslog)")
    print("✓ Kafka streaming")
    print("✓ AWS Lambda (serverless)")

    print("\n4. Chronicle SIEM Integration:")
    print("-" * 70)
    print("Run setup_chronicle.bat to configure Google Chronicle integration")
    print("See GOOGLE_SIEM_SETUP.md for detailed instructions")

    print("\n" + "=" * 70)
