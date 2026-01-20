# Installation Instructions

## Prerequisites

The Log Anomaly Detection Pipeline requires Python 3.8+ with several data science libraries.

## Step 1: Install Python

If you don't have Python installed:

### Windows
1. Download Python from https://www.python.org/downloads/
2. Run the installer and **check "Add Python to PATH"**
3. Verify installation:
   ```cmd
   python --version
   ```

### Alternative: Use Anaconda (Recommended for Data Science)
1. Download Anaconda from https://www.anaconda.com/download
2. Install Anaconda
3. Create a new environment:
   ```cmd
   conda create -n log_anomaly python=3.11
   conda activate log_anomaly
   ```

## Step 2: Install Required Packages

Navigate to the directory containing `requirements.txt`:

```cmd
cd C:\Users\jimmy\.local\bin
```

### Option A: Using pip
```cmd
pip install -r requirements.txt
```

### Option B: Using conda (if using Anaconda)
```cmd
conda install pandas numpy scikit-learn matplotlib seaborn joblib
pip install tensorflow
```

## Step 3: Verify Installation

Run this command to check if all packages are installed:

```cmd
python -c "import pandas, numpy, sklearn, tensorflow, matplotlib, seaborn, joblib; print('All packages installed successfully!')"
```

## Step 4: Test the Pipeline

Once all packages are installed, run the test:

```cmd
python intrusion_detection_pipeline.py --data_path test_logs/ --baseline_period_days 1 --contamination 0.10 --autoencoder_epochs 10
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'X'"
- Install the missing package: `pip install <package_name>`
- Ensure you're using the correct Python environment

### TensorFlow Installation Issues
TensorFlow can be tricky on Windows. If you have issues:

1. **Use CPU version** (works on all systems):
   ```cmd
   pip install tensorflow-cpu
   ```

2. **For GPU support** (requires CUDA):
   - Install CUDA Toolkit and cuDNN
   - Then: `pip install tensorflow`

3. **Alternative**: Use TensorFlow in WSL2 (Windows Subsystem for Linux)

### Memory Issues
If you get memory errors:
- Reduce `--batch_size` (e.g., `--batch_size 256`)
- Reduce `--autoencoder_epochs` (e.g., `--autoencoder_epochs 5`)
- Process smaller log files

### Path Issues on Windows
If you get path errors, use forward slashes or raw strings:
```cmd
python intrusion_detection_pipeline.py --data_path "test_logs/" --output_dir "anomaly_outputs/"
```

## Quick Start After Installation

1. **Test with sample data**:
   ```cmd
   python intrusion_detection_pipeline.py --data_path test_logs/ --baseline_period_days 1 --contamination 0.10 --autoencoder_epochs 10
   ```

2. **Check outputs**:
   ```cmd
   dir anomaly_outputs
   ```

3. **View results**:
   - Open `anomaly_outputs/anomalies_detected.csv` in Excel
   - Open `anomaly_outputs/anomaly_analysis.png` to view visualizations

## Estimated Installation Time
- Downloading Python: 5 minutes
- Installing packages: 10-15 minutes (TensorFlow is large ~500MB)
- Total: ~20 minutes

## System Requirements
- **OS**: Windows 10/11, macOS, Linux
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Disk**: ~2GB for Python + packages
- **CPU**: Any modern CPU (GPU optional for TensorFlow)

## Need Help?
If you encounter issues:
1. Check the error message carefully
2. Ensure Python is in your PATH
3. Try using a virtual environment
4. Consider using Anaconda for easier package management
