#!/usr/bin/env python3
"""
REST API for Log Anomaly Detection
FastAPI server for integration with any platform
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import joblib
import pandas as pd
from io import StringIO

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("FastAPI not installed. Install with: pip install fastapi uvicorn python-multipart")
    exit(1)

# Import anomaly detection components
from log_anomaly_detection_lite import (
    LogParser,
    LogFeaturePipeline,
    StatisticalAnomalyDetector,
    AnomalyScorer,
    preprocess_logs
)
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anomaly-api")

# Initialize FastAPI
app = FastAPI(
    title="Log Anomaly Detection API",
    description="AI-powered security threat detection for system logs",
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
    return_all_events: bool = Field(False, description="Return all events with scores, not just anomalies")


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


class ModelInfo(BaseModel):
    """Model information."""
    loaded: bool
    loaded_at: Optional[str]
    threshold: Optional[float]
    n_features: Optional[int]
    feature_names: Optional[List[str]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    timestamp: str


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


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=MODEL_STATE["loaded"],
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/models/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded models."""
    if not MODEL_STATE["loaded"]:
        return ModelInfo(loaded=False, loaded_at=None, threshold=None, n_features=None, feature_names=None)

    return ModelInfo(
        loaded=True,
        loaded_at=MODEL_STATE["loaded_at"],
        threshold=float(MODEL_STATE["threshold"]) if MODEL_STATE["threshold"] else None,
        n_features=len(MODEL_STATE["feature_pipeline"].feature_names_),
        feature_names=MODEL_STATE["feature_pipeline"].feature_names_
    )


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

        return {
            "status": "success",
            "message": f"Models loaded from {model_dir}",
            "loaded_at": MODEL_STATE["loaded_at"]
        }

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_logs(request: AnalysisRequest):
    """Analyze logs for anomalies."""
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
                processing_time_ms=0.0
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
        for _, row in result_df.iterrows():
            if request.return_all_events or row.get('is_anomaly', True):
                anomalies.append(Anomaly(
                    timestamp=str(row['timestamp']),
                    user=row['user'],
                    source_ip=row['source_ip'],
                    dest_ip=row.get('dest_ip', 'unknown'),
                    event_type=row['event_type'],
                    action=row['action'],
                    message=row['message'],
                    severity=row['severity'],
                    anomaly_score=float(row['anomaly_score']),
                    threat_type=row['threat_type']
                ))

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return AnalysisResponse(
            status="success",
            total_events=len(df),
            anomalies_detected=int(is_anomaly.sum()),
            anomaly_rate=float(is_anomaly.sum() / len(df)),
            threshold=float(threshold),
            anomalies=anomalies if not request.return_all_events else [a for a in anomalies if a.anomaly_score > threshold],
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
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
        request = AnalysisRequest(logs=log_events)
        return await analyze_logs(request)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"File analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    """Get system statistics."""
    if not MODEL_STATE["loaded"]:
        raise HTTPException(status_code=400, detail="Models not loaded")

    return {
        "models_loaded": True,
        "loaded_at": MODEL_STATE["loaded_at"],
        "threshold": float(MODEL_STATE["threshold"]),
        "isolation_forest": {
            "n_estimators": MODEL_STATE["isolation_forest"].n_estimators,
            "contamination": MODEL_STATE["isolation_forest"].contamination
        },
        "features": {
            "count": len(MODEL_STATE["feature_pipeline"].feature_names_),
            "names": MODEL_STATE["feature_pipeline"].feature_names_
        }
    }


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
        except Exception as e:
            logger.warning(f"Failed to auto-load models: {e}")

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
