#!/usr/bin/env python3
"""
REST API for Log Anomaly Detection with Google Chronicle Integration
FastAPI server with automatic Chronicle SIEM forwarding
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import joblib
import pandas as pd
from io import StringIO

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import anomaly detection components
from log_anomaly_detection_lite import (
    LogParser,
    LogFeaturePipeline,
    StatisticalAnomalyDetector,
    AnomalyScorer,
    preprocess_logs
)
from sklearn.ensemble import IsolationForest

# Import Chronicle integration
from google_chronicle_integration import ChronicleClient, ChronicleConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anomaly-api-chronicle")

# Initialize FastAPI
app = FastAPI(
    title="Log Anomaly Detection API with Google Chronicle",
    description="AI-powered security threat detection with Google SIEM integration",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
MODEL_STATE = {
    "feature_pipeline": None,
    "isolation_forest": None,
    "statistical_detector": None,
    "scorer": None,
    "threshold": None,
    "loaded": False,
    "loaded_at": None
}

CHRONICLE_STATE = {
    "client": None,
    "enabled": False,
    "config": None
}


# Pydantic Models
class LogEvent(BaseModel):
    """Single log event."""
    timestamp: str
    user: str
    source_ip: str
    dest_ip: Optional[str] = "unknown"
    event_type: str
    action: str
    message: str
    severity: Optional[str] = "low"


class AnalysisRequest(BaseModel):
    """Request for log analysis."""
    logs: List[LogEvent] = Field(..., description="List of log events to analyze")
    return_all_events: bool = Field(False, description="Return all events with scores")
    send_to_chronicle: bool = Field(True, description="Automatically send anomalies to Chronicle")


class Anomaly(BaseModel):
    """Detected anomaly."""
    timestamp: str
    user: str
    source_ip: str
    dest_ip: str
    event_type: str
    action: str
    message: str
    severity: str
    anomaly_score: float
    threat_type: str


class AnalysisResponse(BaseModel):
    """Analysis results."""
    status: str
    total_events: int
    anomalies_detected: int
    anomaly_rate: float
    threshold: float
    anomalies: List[Anomaly]
    processing_time_ms: float
    chronicle_sent: Optional[bool] = None


class ChronicleStatus(BaseModel):
    """Chronicle integration status."""
    enabled: bool
    configured: bool
    customer_id: Optional[str] = None
    region: Optional[str] = None
    events_sent_session: int = 0


# Utility Functions
def classify_threat(row: pd.Series) -> str:
    """Classify threat type."""
    if 'failed' in str(row.get('action', '')).lower():
        return 'brute_force'
    elif 'sudo' in str(row.get('message', '')).lower():
        return 'privilege_escalation'
    elif any(word in str(row.get('message', '')).lower() for word in ['shadow', 'passwd', 'secret']):
        return 'data_exfiltration'
    elif row.get('event_type') == 'network':
        return 'lateral_movement'
    else:
        return 'unknown'


def assign_severity(score: float) -> str:
    """Assign severity based on score."""
    if score >= 0.95:
        return 'critical'
    elif score >= 0.85:
        return 'high'
    elif score >= 0.7:
        return 'medium'
    else:
        return 'low'


# Chronicle Integration
async def send_to_chronicle_background(anomalies: List[Dict[str, Any]]):
    """Background task to send anomalies to Chronicle."""
    if not CHRONICLE_STATE["enabled"] or not CHRONICLE_STATE["client"]:
        logger.debug("Chronicle integration not enabled")
        return

    try:
        result = CHRONICLE_STATE["client"].send_anomalies(anomalies)
        if result['status'] == 'success':
            logger.info(f"Sent {len(anomalies)} anomalies to Chronicle")
        else:
            logger.error(f"Chronicle ingestion failed: {result.get('message')}")
    except Exception as e:
        logger.error(f"Error sending to Chronicle: {e}")


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": MODEL_STATE["loaded"],
        "chronicle_enabled": CHRONICLE_STATE["enabled"],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/chronicle/status", response_model=ChronicleStatus)
async def chronicle_status():
    """Get Chronicle integration status."""
    return ChronicleStatus(
        enabled=CHRONICLE_STATE["enabled"],
        configured=CHRONICLE_STATE["client"] is not None,
        customer_id=CHRONICLE_STATE["config"].get("customer_id") if CHRONICLE_STATE["config"] else None,
        region=CHRONICLE_STATE["config"].get("region") if CHRONICLE_STATE["config"] else None
    )


@app.post("/chronicle/enable")
async def enable_chronicle(
    credentials_file: str = "chronicle_credentials.json",
    customer_id: str = None,
    region: str = "us"
):
    """Enable Chronicle integration."""
    try:
        # Load config
        config = ChronicleConfig()

        # Use provided params or config
        creds = credentials_file or config.get("credentials_file")
        cust_id = customer_id or config.get("customer_id")
        reg = region or config.get("region", "us")

        # Validate credentials file exists
        if not Path(creds).exists():
            raise HTTPException(
                status_code=400,
                detail=f"Credentials file not found: {creds}"
            )

        # Initialize Chronicle client
        chronicle_client = ChronicleClient(
            credentials_file=creds,
            customer_id=cust_id,
            region=reg
        )

        CHRONICLE_STATE["client"] = chronicle_client
        CHRONICLE_STATE["enabled"] = True
        CHRONICLE_STATE["config"] = {
            "customer_id": cust_id,
            "region": reg,
            "credentials_file": creds
        }

        logger.info("Chronicle integration enabled")

        return {
            "status": "success",
            "message": "Chronicle integration enabled",
            "customer_id": cust_id,
            "region": reg
        }

    except Exception as e:
        logger.error(f"Failed to enable Chronicle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chronicle/disable")
async def disable_chronicle():
    """Disable Chronicle integration."""
    CHRONICLE_STATE["enabled"] = False
    CHRONICLE_STATE["client"] = None
    return {
        "status": "success",
        "message": "Chronicle integration disabled"
    }


@app.post("/chronicle/test")
async def test_chronicle():
    """Test Chronicle connection."""
    if not CHRONICLE_STATE["enabled"]:
        raise HTTPException(
            status_code=400,
            detail="Chronicle integration not enabled. Call POST /chronicle/enable first."
        )

    # Send test event
    test_anomaly = {
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "user": "test_user",
        "source_ip": "192.168.1.100",
        "dest_ip": "unknown",
        "event_type": "login",
        "action": "failed",
        "message": "Test event from API",
        "severity": "low",
        "anomaly_score": 0.65,
        "threat_type": "brute_force"
    }

    try:
        result = CHRONICLE_STATE["client"].send_anomalies([test_anomaly])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/load")
async def load_models(model_dir: str = "anomaly_outputs"):
    """Load trained models from disk."""
    try:
        model_path = Path(model_dir)

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model directory not found: {model_dir}")

        MODEL_STATE["feature_pipeline"] = joblib.load(model_path / "feature_pipeline.pkl")
        MODEL_STATE["isolation_forest"] = joblib.load(model_path / "isolation_forest_model.pkl")
        MODEL_STATE["statistical_detector"] = joblib.load(model_path / "statistical_detector.pkl")

        # Load inference package
        if (model_path / "inference_package.pkl").exists():
            package = joblib.load(model_path / "inference_package.pkl")
            MODEL_STATE["scorer"] = package.get("scorer")
            MODEL_STATE["threshold"] = package.get("threshold")
        else:
            MODEL_STATE["scorer"] = AnomalyScorer()
            MODEL_STATE["threshold"] = 0.7

        MODEL_STATE["loaded"] = True
        MODEL_STATE["loaded_at"] = datetime.utcnow().isoformat()

        logger.info(f"Models loaded successfully from {model_dir}")

        # Try to auto-enable Chronicle if configured
        try:
            config = ChronicleConfig()
            if Path(config.get("credentials_file")).exists():
                await enable_chronicle()
        except:
            logger.info("Chronicle auto-enable skipped (not configured)")

        return {
            "status": "success",
            "message": f"Models loaded from {model_dir}",
            "loaded_at": MODEL_STATE["loaded_at"],
            "chronicle_enabled": CHRONICLE_STATE["enabled"]
        }

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_logs(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze logs for anomalies and optionally send to Chronicle."""
    start_time = datetime.utcnow()

    if not MODEL_STATE["loaded"]:
        raise HTTPException(
            status_code=400,
            detail="Models not loaded. Call POST /models/load first."
        )

    try:
        # Convert request to DataFrame
        logs_dict = [log.dict() for log in request.logs]
        df = pd.DataFrame(logs_dict)

        # Preprocess
        parser = LogParser()
        df = parser._normalize_schema(df)
        df = preprocess_logs(df)

        if len(df) == 0:
            return AnalysisResponse(
                status="success",
                total_events=0,
                anomalies_detected=0,
                anomaly_rate=0.0,
                threshold=float(MODEL_STATE["threshold"]),
                anomalies=[],
                processing_time_ms=0.0,
                chronicle_sent=False
            )

        # Extract features
        features = MODEL_STATE["feature_pipeline"].transform(df)

        # Detect anomalies
        iso_scores = -MODEL_STATE["isolation_forest"].score_samples(features)
        stat_scores = MODEL_STATE["statistical_detector"].detect_all(df)

        # Combine scores
        combined_scores = MODEL_STATE["scorer"].combine_scores({
            'isolation_forest': iso_scores,
            'statistical': stat_scores
        })

        # Identify anomalies
        threshold = MODEL_STATE["threshold"]
        is_anomaly = combined_scores > threshold

        # Build response
        if request.return_all_events:
            result_df = df.copy()
            result_df['anomaly_score'] = combined_scores
            result_df['is_anomaly'] = is_anomaly
        else:
            result_df = df[is_anomaly].copy()
            result_df['anomaly_score'] = combined_scores[is_anomaly]

        # Classify threats and assign severity
        result_df['threat_type'] = result_df.apply(classify_threat, axis=1)
        result_df['severity'] = result_df['anomaly_score'].apply(assign_severity)

        # Convert to response format
        anomalies = []
        anomalies_for_chronicle = []

        for _, row in result_df.iterrows():
            anomaly_dict = {
                "timestamp": str(row['timestamp']),
                "user": row['user'],
                "source_ip": row['source_ip'],
                "dest_ip": row.get('dest_ip', 'unknown'),
                "event_type": row['event_type'],
                "action": row['action'],
                "message": row['message'],
                "severity": row['severity'],
                "anomaly_score": float(row['anomaly_score']),
                "threat_type": row['threat_type']
            }

            if request.return_all_events or row.get('is_anomaly', True):
                anomalies.append(Anomaly(**anomaly_dict))

                # Collect for Chronicle
                if row['anomaly_score'] > threshold:
                    anomalies_for_chronicle.append(anomaly_dict)

        # Send to Chronicle in background if enabled
        chronicle_sent = False
        if request.send_to_chronicle and anomalies_for_chronicle and CHRONICLE_STATE["enabled"]:
            background_tasks.add_task(send_to_chronicle_background, anomalies_for_chronicle)
            chronicle_sent = True

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return AnalysisResponse(
            status="success",
            total_events=len(df),
            anomalies_detected=int(is_anomaly.sum()),
            anomaly_rate=float(is_anomaly.sum() / len(df)),
            threshold=float(threshold),
            anomalies=anomalies if not request.return_all_events else [a for a in anomalies if a.anomaly_score > threshold],
            processing_time_ms=processing_time,
            chronicle_sent=chronicle_sent
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    send_to_chronicle: bool = True,
    background_tasks: BackgroundTasks = None
):
    """Analyze logs from uploaded file."""
    if not MODEL_STATE["loaded"]:
        raise HTTPException(
            status_code=400,
            detail="Models not loaded. Call POST /models/load first."
        )

    try:
        # Read file
        content = await file.read()
        content_str = content.decode('utf-8')

        # Parse based on file extension
        if file.filename.endswith('.json'):
            logs = json.loads(content_str)
            if isinstance(logs, dict):
                logs = [logs]
            df = pd.DataFrame(logs)
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(StringIO(content_str))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use JSON or CSV.")

        # Convert to request format
        logs_list = df.to_dict(orient='records')
        log_events = [LogEvent(**log) for log in logs_list]

        # Use existing analyze endpoint
        request = AnalysisRequest(
            logs=log_events,
            send_to_chronicle=send_to_chronicle
        )
        return await analyze_logs(request, background_tasks)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"File analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Auto-load models if available
    if Path("anomaly_outputs").exists():
        try:
            MODEL_STATE["feature_pipeline"] = joblib.load("anomaly_outputs/feature_pipeline.pkl")
            MODEL_STATE["isolation_forest"] = joblib.load("anomaly_outputs/isolation_forest_model.pkl")
            MODEL_STATE["statistical_detector"] = joblib.load("anomaly_outputs/statistical_detector.pkl")

            if Path("anomaly_outputs/inference_package.pkl").exists():
                package = joblib.load("anomaly_outputs/inference_package.pkl")
                MODEL_STATE["scorer"] = package.get("scorer")
                MODEL_STATE["threshold"] = package.get("threshold")
            else:
                MODEL_STATE["scorer"] = AnomalyScorer()
                MODEL_STATE["threshold"] = 0.7

            MODEL_STATE["loaded"] = True
            MODEL_STATE["loaded_at"] = datetime.utcnow().isoformat()
            logger.info("Models auto-loaded successfully")

            # Try to auto-enable Chronicle
            try:
                config = ChronicleConfig()
                if Path(config.get("credentials_file")).exists():
                    chronicle_client = ChronicleClient(
                        credentials_file=config.get("credentials_file"),
                        customer_id=config.get("customer_id"),
                        region=config.get("region", "us")
                    )
                    CHRONICLE_STATE["client"] = chronicle_client
                    CHRONICLE_STATE["enabled"] = True
                    CHRONICLE_STATE["config"] = config.config
                    logger.info("Chronicle integration auto-enabled")
            except:
                logger.info("Chronicle auto-enable skipped (not configured)")

        except Exception as e:
            logger.warning(f"Failed to auto-load models: {e}")

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
