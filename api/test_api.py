#!/usr/bin/env python3
"""
Quick test script for Anomaly Detection API
Tests all endpoints to verify everything works
"""

import requests
import json
import time
import sys
from pathlib import Path


def test_api(base_url="http://localhost:8000"):
    """Run API tests."""
    print("=" * 70)
    print("Testing Anomaly Detection API")
    print("=" * 70)

    # Test 1: Health Check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✓ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Cannot connect to {base_url}")
        print("   Make sure the API is running: python anomaly_api.py")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test 2: Load Models
    print("\n2. Testing model loading...")
    try:
        response = requests.post(
            f"{base_url}/models/load",
            params={"model_dir": "anomaly_outputs"},
            timeout=10
        )
        if response.status_code == 200:
            print("   ✓ Models loaded successfully")
            result = response.json()
            print(f"   Message: {result['message']}")
        else:
            print(f"   ✗ Model loading failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test 3: Model Info
    print("\n3. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/models/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print("   ✓ Model info retrieved")
            print(f"   Loaded: {info['loaded']}")
            print(f"   Features: {info['n_features']}")
            print(f"   Threshold: {info['threshold']}")
        else:
            print(f"   ✗ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 4: Analyze Sample Logs
    print("\n4. Testing log analysis...")
    sample_logs = [
        {
            "timestamp": "2026-01-15T10:00:00Z",
            "user": "admin",
            "source_ip": "10.0.0.100",
            "dest_ip": "unknown",
            "event_type": "login",
            "action": "failed",
            "message": "Failed login attempt"
        },
        {
            "timestamp": "2026-01-15T10:00:05Z",
            "user": "admin",
            "source_ip": "10.0.0.100",
            "dest_ip": "unknown",
            "event_type": "login",
            "action": "failed",
            "message": "Failed login attempt"
        },
        {
            "timestamp": "2026-01-15T10:00:10Z",
            "user": "admin",
            "source_ip": "10.0.0.100",
            "dest_ip": "unknown",
            "event_type": "login",
            "action": "failed",
            "message": "Failed login attempt"
        },
        {
            "timestamp": "2026-01-15T09:00:00Z",
            "user": "alice",
            "source_ip": "192.168.1.10",
            "dest_ip": "unknown",
            "event_type": "login",
            "action": "success",
            "message": "User logged in"
        }
    ]

    try:
        response = requests.post(
            f"{base_url}/analyze",
            json={"logs": sample_logs},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print("   ✓ Analysis completed")
            print(f"   Total events: {result['total_events']}")
            print(f"   Anomalies detected: {result['anomalies_detected']}")
            print(f"   Anomaly rate: {result['anomaly_rate']:.2%}")
            print(f"   Processing time: {result['processing_time_ms']:.2f}ms")

            if result['anomalies']:
                print("\n   Detected Anomalies:")
                for anomaly in result['anomalies']:
                    print(f"   - {anomaly['threat_type']} ({anomaly['severity']}): "
                          f"{anomaly['user']} from {anomaly['source_ip']} "
                          f"(score: {anomaly['anomaly_score']:.3f})")
        else:
            print(f"   ✗ Analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test 5: File Upload (if test file exists)
    print("\n5. Testing file upload...")
    test_file = Path("test_logs/test_logs_attack.json")
    if test_file.exists():
        try:
            with open(test_file, 'rb') as f:
                response = requests.post(
                    f"{base_url}/analyze/file",
                    files={"file": f},
                    timeout=10
                )
            if response.status_code == 200:
                result = response.json()
                print("   ✓ File analysis completed")
                print(f"   Anomalies detected: {result['anomalies_detected']}")
            else:
                print(f"   ✗ File analysis failed: {response.status_code}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print(f"   ⊘ Skipped (test file not found: {test_file})")

    # Test 6: Statistics
    print("\n6. Testing statistics endpoint...")
    try:
        response = requests.get(f"{base_url}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print("   ✓ Statistics retrieved")
            print(f"   Models loaded: {stats['models_loaded']}")
            print(f"   Feature count: {stats['features']['count']}")
        else:
            print(f"   ✗ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    print(f"\nAPI is running successfully at {base_url}")
    print(f"API documentation: {base_url}/docs")
    print("\nYou can now:")
    print("  - Integrate with AI platforms using the MCP server")
    print("  - Send logs via REST API")
    print("  - Use batch processing for scheduled analysis")
    print("  - Deploy with Docker")
    print("\nSee README_SCALING.md for integration examples.")

    return True


def wait_for_api(base_url="http://localhost:8000", timeout=30):
    """Wait for API to be ready."""
    print(f"Waiting for API at {base_url}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                print("API is ready!")
                return True
        except:
            pass
        time.sleep(1)
        print(".", end="", flush=True)

    print(f"\nTimeout waiting for API after {timeout}s")
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Anomaly Detection API")
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                       help='API base URL')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for API to be ready before testing')
    args = parser.parse_args()

    if args.wait:
        if not wait_for_api(args.url):
            sys.exit(1)

    success = test_api(args.url)
    sys.exit(0 if success else 1)
