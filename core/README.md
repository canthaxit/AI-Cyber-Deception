# Core Detection Engine

Main anomaly detection algorithms and models.

## Files

- **log_anomaly_detection_lite.py** - Lightweight version (no TensorFlow)
- **intrusion_detection_pipeline.py** - Full version with neural networks
- **anomaly_outputs/** - Trained models (created after first run)

## Quick Start

```bash
# Train models with test data
python log_anomaly_detection_lite.py \
  --data_path ../tests/test_logs_*.json \
  --baseline_period_days 1 \
  --output_dir anomaly_outputs

# Analyze your logs
python log_anomaly_detection_lite.py \
  --data_path /path/to/your/logs/ \
  --output_dir anomaly_outputs
```

## Output Files

After running, the `anomaly_outputs/` directory contains:
- `feature_pipeline.pkl` - Feature extraction pipeline
- `isolation_forest_model.pkl` - Isolation Forest model
- `statistical_detector.pkl` - Statistical rule detector
- `inference_package.pkl` - Complete inference package
- `anomalies_detected.csv` - Detected anomalies (CSV)
- `anomalies_detailed.json` - Detailed anomaly report (JSON)
- `anomaly_analysis.png` - Visualization plots

## Usage from Other Modules

```python
import sys
sys.path.insert(0, '/path/to/anomaly-detection/core')

from log_anomaly_detection_lite import (
    LogParser,
    LogFeaturePipeline,
    StatisticalAnomalyDetector
)

# Use the modules...
```

## Documentation

See [`../docs/LITE_VERSION_GUIDE.md`](../docs/LITE_VERSION_GUIDE.md) for details.
