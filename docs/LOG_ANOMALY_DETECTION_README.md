# Log Anomaly Detection Pipeline

## Overview
This pipeline has been transformed from a supervised network intrusion detection system to an **unsupervised log anomaly detection system** that detects threat actor patterns in JSON-formatted system and security logs.

## Key Features
- **JSON Log Parsing**: Automatically parses and normalizes JSON log files
- **Unsupervised Detection**: No labeled data required - learns normal behavior from baseline period
- **Multi-Model Ensemble**: Combines Isolation Forest, Autoencoder, and Statistical threat detectors
- **Threat Pattern Detection**: Specifically detects:
  - Brute force / credential attacks
  - Privilege escalation attempts
  - Data exfiltration patterns
  - Lateral movement indicators
- **Temporal & Behavioral Features**: Extracts time-based and user/IP behavioral features
- **Complete Reporting**: CSV, JSON reports with visualizations

## Quick Start

### 1. Test with Sample Data
Two test log files have been created:
- `test_logs_normal.json` - Normal baseline behavior (Day 1-7)
- `test_logs_attack.json` - Contains attacks (Day 8)

Create a test directory and run:
```bash
mkdir test_logs
cp test_logs_normal.json test_logs/
cp test_logs_attack.json test_logs/

python intrusion_detection_pipeline.py \
    --data_path test_logs/ \
    --baseline_period_days 1 \
    --contamination 0.10 \
    --autoencoder_epochs 10
```

### 2. Run with Your Own Logs
Place your JSON log files in a directory and run:
```bash
python intrusion_detection_pipeline.py \
    --data_path /path/to/your/logs/ \
    --output_dir ./anomaly_outputs \
    --baseline_period_days 7 \
    --contamination 0.01
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | `./logs/` | Directory containing JSON log files |
| `--log_format` | `auto` | Log format (auto, auth, syslog, security) |
| `--output_dir` | `anomaly_outputs` | Output directory for results |
| `--contamination` | `0.01` | Expected anomaly rate (1% = 0.01) |
| `--baseline_period_days` | `7` | Days of normal behavior for baseline |
| `--autoencoder_epochs` | `50` | Training epochs for autoencoder |
| `--iso_forest_estimators` | `200` | Number of trees in Isolation Forest |
| `--batch_size` | `1024` | Batch size for autoencoder training |
| `--random_state` | `42` | Random seed for reproducibility |

## JSON Log Format

Your logs should be in JSON format with these fields (flexible field names supported):

```json
{
  "timestamp": "2026-01-15T10:30:00Z",
  "user": "alice",
  "source_ip": "192.168.1.10",
  "event_type": "login",
  "action": "success",
  "message": "User logged in successfully"
}
```

Supported field variations:
- **timestamp**: `timestamp`, `time`, `@timestamp`, `datetime`, `date`
- **user**: `user`, `username`, `uid`, `account`, `identity`
- **source_ip**: `source_ip`, `src_ip`, `ip`, `client_ip`, `remote_addr`
- **event_type**: `event_type`, `type`, `event`, `category`
- **action**: `action`, `result`, `status`, `outcome`
- **message**: `message`, `msg`, `description`, `text`, `log`

## Output Files

After running, check the `anomaly_outputs/` directory for:

1. **anomalies_detected.csv** - Detected anomalies in CSV format
2. **anomalies_detailed.json** - Full anomaly details in JSON
3. **anomaly_analysis.png** - Visualization dashboard with 4 plots:
   - Anomaly scores over time
   - Score distribution histogram
   - Threat type breakdown
   - Top anomalous users
4. **feature_pipeline.pkl** - Trained feature extraction pipeline
5. **isolation_forest_model.pkl** - Trained Isolation Forest
6. **autoencoder_model.keras** - Trained Autoencoder
7. **statistical_detector.pkl** - Statistical threat detector
8. **inference_package.pkl** - Complete package for deployment

## Understanding Results

### Threat Types
- **brute_force**: Multiple failed login attempts, password spraying
- **privilege_escalation**: Unauthorized sudo usage, privilege changes
- **data_exfiltration**: Off-hours file access, sensitive file access
- **lateral_movement**: Unusual network connections, credential reuse

### Severity Levels
- **LOW**: Score 0.5-0.7
- **MEDIUM**: Score 0.7-0.85
- **HIGH**: Score 0.85-0.95
- **CRITICAL**: Score 0.95+

## How It Works

1. **Load & Parse**: Loads all JSON logs from directory
2. **Preprocess**: Removes duplicates, validates timestamps
3. **Temporal Split**: Splits into baseline (normal) and analysis periods
4. **Feature Engineering**: Extracts 40+ features:
   - Temporal: hour, day_of_week, is_weekend, is_night, etc.
   - Behavioral: failed_login_ratio, off_hours_activity, etc.
   - Entity-based: user baselines, IP baselines, deviation scores
5. **Train Detectors**:
   - Isolation Forest (general anomalies)
   - Autoencoder (reconstruction-based)
   - Statistical Rules (threat-specific patterns)
6. **Detect Anomalies**: Combines scores from all detectors
7. **Classify Threats**: Assigns threat type and severity
8. **Report**: Generates CSV, JSON, and visualizations

## Tuning Tips

### High False Positive Rate
- Increase `--baseline_period_days` (more data to learn normal behavior)
- Decrease `--contamination` (stricter threshold)
- Check if baseline period actually contains only normal logs

### Missing Real Attacks
- Increase `--contamination` (more lenient threshold)
- Ensure attack patterns are sufficiently different from baseline
- Review threat detection rules in `StatisticalAnomalyDetector`

### Performance Issues
- Reduce `--autoencoder_epochs` for faster training
- Reduce `--iso_forest_estimators` for faster Isolation Forest
- Process logs in batches if dataset is very large

## Next Steps

1. **Test with sample data** to verify installation
2. **Tune parameters** based on your environment
3. **Review detected anomalies** and refine thresholds
4. **Deploy for real-time monitoring** using the saved artifacts
5. **Integrate with SIEM** or alerting system

## Troubleshooting

### "No JSON files found"
- Ensure JSON files have `.json` extension
- Verify `--data_path` points to correct directory

### "No baseline data available"
- Increase `--baseline_period_days`
- Verify your logs span enough days

### "No analysis data available"
- Decrease `--baseline_period_days`
- Ensure you have logs beyond the baseline period

### Feature extraction errors
- Check timestamp format (should be ISO 8601)
- Verify JSON structure matches expected schema
- Check for null/missing required fields

## Example: Detecting Brute Force Attack

Given test logs:
- Days 1-7: Normal user activity (baseline)
- Day 8: 12 failed login attempts in 1 minute from suspicious IP

Expected output:
```
Anomalies detected: 12 / 18 (66.67%)

Threat Type Breakdown:
  brute_force: 12
  data_exfiltration: 2
  privilege_escalation: 1

Top Affected Users:
  admin: 12
  root: 1
```

## Support

For issues or questions:
1. Check this README
2. Review test examples
3. Examine console output for errors
4. Verify input JSON format

Happy threat hunting! 🔍
