# Quick Start Guide

## Current Status

✅ **Pipeline Transformed**: Successfully converted to log anomaly detection
✅ **Test Data Created**: Sample logs ready in `test_logs/`
❌ **Dependencies Missing**: Need to install Python packages

## Problem Found

Your system doesn't have a working Python installation with `pip`. The Python executable found (from lmstudio) doesn't include pip for package management.

## Solution: Install Python Properly

### Step 1: Install Python

**Windows:**
1. Go to https://www.python.org/downloads/
2. Download **Python 3.11** (or 3.10/3.12)
3. Run the installer
4. **IMPORTANT**: Check the box "Add Python to PATH" ✓
5. Click "Install Now"

**Verify Installation:**
```cmd
python --version
```
Should show: `Python 3.11.x` or similar

### Step 2: Install Dependencies

**Option A: Automatic (Recommended)**

Run the installer script:
```cmd
cd C:\Users\jimmy\.local\bin
install_dependencies.bat
```

**Option B: Manual**
```cmd
cd C:\Users\jimmy\.local\bin
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This will install:
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (machine learning)
- tensorflow (deep learning)
- matplotlib (plotting)
- seaborn (visualization)
- joblib (model persistence)

**⏱ Estimated Time**: 10-15 minutes (TensorFlow is ~500MB)

### Step 3: Run the Test

Once dependencies are installed:

```cmd
cd C:\Users\jimmy\.local\bin
python intrusion_detection_pipeline.py --data_path test_logs/ --baseline_period_days 1 --contamination 0.10 --autoencoder_epochs 10
```

### Expected Output

```
============================================================
LOG ANOMALY DETECTION PIPELINE
Started at: 2026-01-15 XX:XX:XX
============================================================

LOG LOADING
============================================================
Searching for JSON log files in test_logs/...
  Loading test_logs_attack.json...
    Shape: (18, 7)
  Loading test_logs_normal.json...
    Shape: (10, 7)

Total logs loaded: 28
Time range: 2026-01-01 09:00:00 to 2026-01-08 15:00:00
Memory usage: 0.01 MB

LOG PREPROCESSING
============================================================
[... preprocessing steps ...]

TEMPORAL SPLIT
============================================================
Baseline period: 10 events (2026-01-01 to 2026-01-01)
Analysis period: 18 events (2026-01-08 to 2026-01-08)

TRAINING ANOMALY DETECTORS
============================================================
Training Isolation Forest...
Training Autoencoder...
[... training progress ...]

DETECTING ANOMALIES
============================================================
Anomalies detected: 12-15 / 18 (66-83%)

DETECTION SUMMARY
============================================================
Total events analyzed: 18
Anomalies detected: 12 (66.67%)

Threat Type Breakdown:
  brute_force: 12
  data_exfiltration: 1
  privilege_escalation: 1

Severity Distribution:
  MEDIUM: 8
  HIGH: 4

Top 5 Affected Users:
  admin: 12
  root: 1

All outputs saved to: anomaly_outputs/
```

### Step 4: View Results

**Check outputs:**
```cmd
dir anomaly_outputs
```

**Open visualizations:**
- `anomaly_outputs\anomaly_analysis.png` - View in any image viewer
- `anomaly_outputs\anomalies_detected.csv` - Open in Excel
- `anomaly_outputs\anomalies_detailed.json` - View in text editor

## Troubleshooting

### "Python is not recognized..."
- Python not installed or not in PATH
- **Solution**: Reinstall Python and check "Add to PATH"

### "No module named 'pip'"
- Using wrong Python (like lmstudio's Python)
- **Solution**: Install proper Python from python.org

### TensorFlow install fails
- Try CPU version: `pip install tensorflow-cpu`
- Or skip it temporarily and test basic features

### Still having issues?
- Use Anaconda instead: https://www.anaconda.com/download
- Creates isolated environment with all packages pre-configured

## Alternative: Use Anaconda (Easier)

If you prefer a simpler setup:

1. Download Anaconda: https://www.anaconda.com/download
2. Install Anaconda
3. Open Anaconda Prompt
4. Create environment:
   ```cmd
   conda create -n log_anomaly python=3.11
   conda activate log_anomaly
   conda install pandas numpy scikit-learn matplotlib seaborn joblib
   pip install tensorflow
   ```
5. Run pipeline:
   ```cmd
   cd C:\Users\jimmy\.local\bin
   python intrusion_detection_pipeline.py --data_path test_logs/ --baseline_period_days 1 --contamination 0.10 --autoencoder_epochs 10
   ```

## Files Created for You

```
C:\Users\jimmy\.local\bin\
├── intrusion_detection_pipeline.py       ← Main pipeline
├── test_logs/
│   ├── test_logs_normal.json            ← Baseline data
│   └── test_logs_attack.json            ← Attack scenarios
├── requirements.txt                      ← Package list
├── install_dependencies.bat              ← Windows installer
├── install_dependencies.sh               ← Linux/Mac installer
├── INSTALLATION.md                       ← Detailed setup guide
├── LOG_ANOMALY_DETECTION_README.md       ← Usage guide
└── QUICK_START.md                        ← This file
```

## Next Steps

1. ✅ Install Python with pip
2. ✅ Run `install_dependencies.bat`
3. ✅ Test with sample data
4. ✅ Review results in `anomaly_outputs/`
5. ✅ Use with your own JSON logs

## Need Help?

If you're stuck, you can:
1. Check `INSTALLATION.md` for detailed instructions
2. Try using Anaconda (easier for beginners)
3. Share the error message for troubleshooting

Once Python is properly installed, everything else is automated! 🚀
