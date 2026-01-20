# Batch Processing

Scheduled batch analysis of log files.

## Files

- **batch_processor.py** - Batch processing engine
- **batch_outputs/** - Results directory (created at runtime)

## Quick Start

### One-Time Processing
```bash
python batch_processor.py \
  --log-dir ../tests/ \
  --output-dir ./batch_outputs/ \
  --once
```

### Continuous Processing
```bash
python batch_processor.py \
  --log-dir ../tests/ \
  --output-dir ./batch_outputs/ \
  --interval 3600  # Every hour
```

## How It Works

1. **Scans** the log directory for new JSON/CSV files
2. **Analyzes** each file for anomalies
3. **Saves** results to output directory
4. **Tracks** processed files to avoid duplicates
5. **Repeats** at configured interval

## Output Files

For each processed log file:
- `{filename}_{timestamp}.json` - Detailed anomaly report
- `batch_summary_{timestamp}.json` - Batch processing summary
- `processed_files.txt` - List of processed files

## Example Output

```json
{
  "source_file": "../tests/auth.json",
  "processed_at": "2026-01-15T10:30:00",
  "total_events": 150,
  "anomalies_detected": 8,
  "anomaly_rate": 0.053,
  "threshold": 0.7,
  "anomalies": [
    {
      "timestamp": "2026-01-15T10:00:00Z",
      "user": "admin",
      "source_ip": "10.0.0.100",
      "threat_type": "brute_force",
      "severity": "high",
      "anomaly_score": 0.92
    }
  ]
}
```

## Scheduled Execution

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Daily or Hourly
4. Action: Start a Program
5. Program: `python`
6. Arguments: `C:\path\to\batch_processor.py --once --log-dir C:\logs`

### Linux/Mac Cron

```bash
# Edit crontab
crontab -e

# Run every hour
0 * * * * cd /path/to/anomaly-detection/batch && python batch_processor.py --once

# Run every day at 2 AM
0 2 * * * cd /path/to/anomaly-detection/batch && python batch_processor.py --once
```

### Docker

```bash
docker run -d \
  -v /path/to/logs:/app/logs \
  -v /path/to/outputs:/app/batch_outputs \
  -v /path/to/models:/app/models \
  -e BATCH_INTERVAL=3600 \
  anomaly-detection \
  python batch_processor.py
```

## Integration with SIEM

Send results to Chronicle, Splunk, etc.:

```python
from batch_processor import BatchProcessor
from google_chronicle_integration import ChronicleClient

# Initialize
processor = BatchProcessor(
    log_dir="../tests/",
    output_dir="./batch_outputs/"
)

chronicle = ChronicleClient(
    credentials_file="../chronicle/chronicle_credentials.json",
    customer_id="C00000000"
)

# Process batch
processor.load_models()
result = processor.process_batch()

# Send to Chronicle
for file_result in result['results']:
    if file_result['status'] == 'success':
        import json
        with open(file_result['output_file']) as f:
            data = json.load(f)
            chronicle.send_anomalies(data['anomalies'])
```

## Configuration

### Command Line Options

```bash
python batch_processor.py --help

Options:
  --model-dir PATH      Path to trained models (default: ../core/anomaly_outputs)
  --log-dir PATH        Path to log files (default: logs/)
  --output-dir PATH     Path to save results (default: batch_outputs/)
  --interval SECONDS    Processing interval (default: 3600)
  --once               Run once and exit (default: continuous)
```

### Environment Variables

```bash
export MODEL_DIR=/path/to/models
export LOG_DIR=/path/to/logs
export OUTPUT_DIR=/path/to/outputs
export BATCH_INTERVAL=3600

python batch_processor.py
```

## Monitoring

### Check Processed Files

```bash
# List processed files
cat batch_outputs/processed_files.txt

# Count processed files
wc -l batch_outputs/processed_files.txt

# View latest summary
ls -t batch_outputs/batch_summary_*.json | head -1 | xargs cat
```

### Statistics

```bash
# Count total anomalies detected
jq '.anomalies_detected' batch_outputs/*.json | awk '{s+=$1} END {print s}'

# Find files with high anomaly rates
jq -r 'select(.anomaly_rate > 0.1) | .source_file' batch_outputs/*.json
```

## Performance

### Throughput
- **Single file**: ~500-1000 events/second
- **Batch mode**: ~10,000 events/second
- **Memory usage**: ~500MB per worker

### Optimization

**Process large files efficiently:**
```python
# Custom batch size
processor = BatchProcessor(
    log_dir="logs/",
    output_dir="batch_outputs/"
)

# Process in chunks
import pandas as pd

for chunk in pd.read_json("large_file.json", lines=True, chunksize=10000):
    # Process chunk
    pass
```

## Error Handling

The processor handles errors gracefully:
- **Invalid JSON**: Logged and skipped
- **Missing models**: Error logged, processing continues for other files
- **Disk space**: Warnings on low space
- **File permissions**: Logged and skipped

Check errors:
```bash
# View batch summary for errors
jq -r 'select(.results[].status == "error")' batch_outputs/batch_summary_*.json
```

## Use Cases

### 1. Daily Security Report
```bash
# Run daily at 6 AM, process yesterday's logs
0 6 * * * python batch_processor.py --log-dir /var/log/archive/ --once
```

### 2. Real-Time with Lag
```bash
# Process every 5 minutes for near-real-time
*/5 * * * * python batch_processor.py --log-dir /var/log/current/ --once
```

### 3. Archive Analysis
```bash
# Process historical logs
python batch_processor.py \
  --log-dir /archive/2025/ \
  --output-dir /results/2025/ \
  --once
```

## Documentation

- **Scaling Guide**: [`../docs/SCALING_GUIDE.md`](../docs/SCALING_GUIDE.md)
- **Integration Examples**: [`../examples/integration_examples.py`](../examples/integration_examples.py)
