# Migration Guide - New Project Structure

The project has been reorganized for better maintainability and clarity.

## What Changed

### Before (Old Structure)
```
C:\Users\jimmy\.local\bin\
├── All files mixed together (~40 files)
├── *.py, *.md, *.json, *.bat, *.txt
└── Hard to navigate and maintain
```

### After (New Structure)
```
C:\Users\jimmy\.local\bin\anomaly-detection\
├── core/           # Core detection engine
├── api/            # REST API service
├── mcp/            # MCP server for AI tools
├── chronicle/      # Google Chronicle integration
├── batch/          # Batch processing
├── docker/         # Container deployment
├── tests/          # Test data
├── examples/       # Integration examples
├── config/         # Configuration files
└── docs/           # Documentation
```

## File Locations

| Old Location | New Location |
|-------------|--------------|
| `log_anomaly_detection_lite.py` | `core/log_anomaly_detection_lite.py` |
| `anomaly_api.py` | `api/anomaly_api.py` |
| `anomaly_mcp_server.py` | `mcp/anomaly_mcp_server.py` |
| `google_chronicle_integration.py` | `chronicle/google_chronicle_integration.py` |
| `batch_processor.py` | `batch/batch_processor.py` |
| `Dockerfile` | `docker/Dockerfile` |
| `docker-compose.yml` | `docker/docker-compose.yml` |
| `test_logs_*.json` | `tests/test_logs_*.json` |
| `integration_examples.py` | `examples/integration_examples.py` |
| `requirements*.txt` | `config/requirements*.txt` |
| `*.md` | `docs/*.md` |

## Path Updates Required

### 1. Import Statements

**Old way:**
```python
from log_anomaly_detection_lite import LogParser
```

**New way:**
```python
import sys
sys.path.insert(0, '../core')
from log_anomaly_detection_lite import LogParser
```

Or use absolute paths:
```python
import sys
sys.path.insert(0, 'C:/Users/jimmy/.local/bin/anomaly-detection/core')
from log_anomaly_detection_lite import LogParser
```

### 2. Model Directory

**Old way:**
```python
MODEL_DIR = "anomaly_outputs"
```

**New way:**
```python
# From api directory
MODEL_DIR = "../core/anomaly_outputs"

# Or absolute path
MODEL_DIR = "C:/Users/jimmy/.local/bin/anomaly-detection/core/anomaly_outputs"
```

### 3. Test Data

**Old way:**
```python
parser.load_logs("test_logs/")
```

**New way:**
```python
# From core directory
parser.load_logs("../tests/")

# Or absolute path
parser.load_logs("C:/Users/jimmy/.local/bin/anomaly-detection/tests/")
```

### 4. MCP Configuration

**Old way:**
```json
{
  "args": ["C:\\Users\\jimmy\\.local\\bin\\anomaly_mcp_server.py"],
  "env": {"MODEL_DIR": "anomaly_outputs"}
}
```

**New way:**
```json
{
  "args": ["C:\\Users\\jimmy\\.local\\bin\\anomaly-detection\\mcp\\anomaly_mcp_server.py"],
  "env": {"MODEL_DIR": "C:\\Users\\jimmy\\.local\\bin\\anomaly-detection\\core\\anomaly_outputs"}
}
```

### 5. Docker Volumes

**Old way:**
```yaml
volumes:
  - ./anomaly_outputs:/app/anomaly_outputs
```

**New way:**
```yaml
volumes:
  - ../core/anomaly_outputs:/app/anomaly_outputs
  - ../tests:/app/logs
```

## Quick Migration Steps

### Step 1: Update Environment Variables (Recommended)

**Windows:**
```batch
set ANOMALY_HOME=C:\Users\jimmy\.local\bin\anomaly-detection
set MODEL_DIR=%ANOMALY_HOME%\core\anomaly_outputs
set LOG_DIR=%ANOMALY_HOME%\tests
set CONFIG_DIR=%ANOMALY_HOME%\config
```

Add to system PATH:
```batch
setx ANOMALY_HOME "C:\Users\jimmy\.local\bin\anomaly-detection"
```

**Linux/Mac:**
```bash
export ANOMALY_HOME=/path/to/anomaly-detection
export MODEL_DIR=$ANOMALY_HOME/core/anomaly_outputs
export LOG_DIR=$ANOMALY_HOME/tests
export CONFIG_DIR=$ANOMALY_HOME/config
```

Add to `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export ANOMALY_HOME=/path/to/anomaly-detection' >> ~/.bashrc
```

### Step 2: Update Scripts

**In Python scripts:**
```python
import os

# Use environment variable
ANOMALY_HOME = os.getenv('ANOMALY_HOME', 'C:/Users/jimmy/.local/bin/anomaly-detection')
MODEL_DIR = os.path.join(ANOMALY_HOME, 'core', 'anomaly_outputs')
```

**In batch files:**
```batch
@echo off
if not defined ANOMALY_HOME (
    set ANOMALY_HOME=C:\Users\jimmy\.local\bin\anomaly-detection
)

python %ANOMALY_HOME%\api\anomaly_api.py
```

### Step 3: Retrain Models (If Needed)

If models were in old `anomaly_outputs/`, retrain:
```bash
cd C:\Users\jimmy\.local\bin\anomaly-detection\core
python log_anomaly_detection_lite.py --data_path ../tests/test_logs_*.json
```

### Step 4: Update MCP Config

1. Edit `mcp/claude_mcp_config.json`
2. Update paths to absolute paths
3. Copy to Claude Desktop config location

### Step 5: Test Everything

```bash
# Test core
cd core/
python log_anomaly_detection_lite.py --data_path ../tests/

# Test API
cd ../api/
python anomaly_api.py &
python test_api.py

# Test batch
cd ../batch/
python batch_processor.py --log-dir ../tests/ --once
```

## Common Issues

### Issue: "Module not found"

**Solution:**
```python
import sys
import os

# Add core directory to path
core_dir = os.path.join(os.path.dirname(__file__), '..', 'core')
sys.path.insert(0, os.path.abspath(core_dir))

# Now imports work
from log_anomaly_detection_lite import LogParser
```

### Issue: "Models not found"

**Solution:**
```python
import os

# Get correct path
if os.getenv('MODEL_DIR'):
    MODEL_DIR = os.getenv('MODEL_DIR')
else:
    # Relative to current file
    MODEL_DIR = os.path.join(
        os.path.dirname(__file__),
        '..',
        'core',
        'anomaly_outputs'
    )

# Load models
pipeline = joblib.load(os.path.join(MODEL_DIR, 'feature_pipeline.pkl'))
```

### Issue: "Test data not found"

**Solution:**
```python
import os

# Find test directory
TEST_DIR = os.path.join(
    os.path.dirname(__file__),
    '..',
    'tests'
)

# Load test data
parser.load_logs(os.path.join(TEST_DIR, 'test_logs_normal.json'))
```

## Advantages of New Structure

### ✅ Better Organization
- Related files grouped together
- Clear separation of concerns
- Easier to navigate

### ✅ Easier Deployment
- Each component can be deployed independently
- Clear boundaries between services
- Better for microservices architecture

### ✅ Improved Maintenance
- Changes isolated to relevant directories
- Easier to find and update files
- Better for team collaboration

### ✅ Clearer Documentation
- Each directory has its own README
- Component-specific guides
- Easier to onboard new users

## Backwards Compatibility

To maintain compatibility with old scripts:

### Option 1: Symbolic Links (Windows)

```batch
mklink /D "C:\Users\jimmy\.local\bin\anomaly_outputs" "C:\Users\jimmy\.local\bin\anomaly-detection\core\anomaly_outputs"
mklink "C:\Users\jimmy\.local\bin\anomaly_api.py" "C:\Users\jimmy\.local\bin\anomaly-detection\api\anomaly_api.py"
```

### Option 2: Wrapper Scripts

Create `anomaly_api.py` in old location:
```python
import sys
import os

# Redirect to new location
new_location = os.path.join(
    os.path.dirname(__file__),
    'anomaly-detection',
    'api',
    'anomaly_api.py'
)

# Execute new script
exec(open(new_location).read())
```

### Option 3: Update Scripts

Recommended: Update all scripts to use new paths.

## Need Help?

- **Main README**: [`README.md`](README.md)
- **Quick Start**: [`docs/QUICK_START.md`](docs/QUICK_START.md)
- **Integration Guide**: [`docs/README_INTEGRATIONS.md`](docs/README_INTEGRATIONS.md)

Each directory also has its own README with specific instructions.

## Rollback

If you need to revert to old structure:

```bash
cd C:\Users\jimmy\.local\bin\anomaly-detection

# Move files back to parent
mv core/*.py ..
mv api/*.py ..
mv mcp/*.py ..
mv chronicle/*.py ..
mv batch/*.py ..
mv docker/* ..
mv tests/* ..
mv examples/*.py ..
mv config/* ..
mv docs/*.md ..

cd ..
rm -rf anomaly-detection/
```

---

**Recommendation:** Embrace the new structure. It's cleaner, more maintainable, and follows industry best practices! 🎯
