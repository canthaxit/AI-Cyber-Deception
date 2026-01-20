# 🚀 START HERE - Log Anomaly Detection

Welcome! Your project has been reorganized for better clarity and maintainability.

## ⚡ Quick Start (2 minutes)

### 1. Choose What You Want to Do

```bash
cd C:\Users\jimmy\.local\bin\anomaly-detection
```

Then pick your path:

| I want to... | Run this... |
|--------------|-------------|
| **Analyze logs now** | `cd core && python log_anomaly_detection_lite.py --data_path ../tests/` |
| **Start REST API** | `cd api && start_api.bat` or `python anomaly_api.py` |
| **Use with Claude** | `cd mcp` → See `README.md` for setup |
| **Connect to Chronicle** | `cd chronicle && setup_chronicle.bat` |
| **Run batch processing** | `cd batch && python batch_processor.py --log-dir ../tests/ --once` |
| **Deploy with Docker** | `cd docker && docker-compose up -d` |

### 2. First Time Setup

If you haven't installed dependencies yet:
```bash
pip install -r config/requirements_minimal.txt    # Core only
# OR
pip install -r config/requirements_api.txt        # + API
# OR
pip install -r config/requirements_chronicle.txt  # + Chronicle
```

## 📁 What's Where?

```
anomaly-detection/
├── core/           ← Detection engine (START HERE for analysis)
├── api/            ← REST API service
├── chronicle/      ← Google Chronicle SIEM integration
├── mcp/            ← Claude Desktop integration
├── batch/          ← Scheduled processing
├── docker/         ← Container deployment
├── tests/          ← Sample test data
├── examples/       ← Integration code samples
├── config/         ← Installation & requirements
└── docs/           ← Full documentation
```

## 🎯 Common Tasks

### Analyze Your Logs
```bash
cd core/
python log_anomaly_detection_lite.py --data_path /path/to/your/logs/
```

Results appear in `core/anomaly_outputs/`:
- `anomalies_detected.csv` - Anomalies table
- `anomalies_detailed.json` - Full JSON report
- `anomaly_analysis.png` - Visualization

### Start the API
```bash
cd api/
python anomaly_api.py

# Access at: http://localhost:8000/docs
```

Test it:
```bash
cd api/
python test_api.py
```

### Use with Claude Desktop
```bash
cd mcp/

# Windows
copy claude_mcp_config.json %APPDATA%\Claude\claude_desktop_config.json

# Restart Claude, then ask:
# "Analyze the logs in my tests folder for security threats"
```

### Connect to Google Chronicle
```bash
cd chronicle/
setup_chronicle.bat

# Follow prompts to configure credentials
```

### Process Logs in Batches
```bash
cd batch/
python batch_processor.py \
  --log-dir /path/to/logs/ \
  --output-dir ./batch_outputs/ \
  --interval 3600  # Every hour
```

### Deploy with Docker
```bash
cd docker/
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - MCP: stdio mode
# - Batch: scheduled
```

## 📚 Documentation

Each directory has a `README.md` with details:

| Directory | What It Does | Quick Guide |
|-----------|--------------|-------------|
| [`core/`](core/README.md) | Detect anomalies | Run detection engine |
| [`api/`](api/README.md) | REST API | Start HTTP service |
| [`chronicle/`](chronicle/README.md) | Chronicle SIEM | Google Cloud integration |
| [`mcp/`](mcp/README.md) | AI platforms | Claude Desktop setup |
| [`batch/`](batch/README.md) | Batch jobs | Scheduled processing |
| [`docker/`](docker/README.md) | Containers | Production deployment |
| [`tests/`](tests/README.md) | Test data | Sample logs |
| [`examples/`](examples/README.md) | Code samples | Integration examples |

**Main Docs:**
- [`docs/QUICK_START.md`](docs/QUICK_START.md) - Detailed getting started
- [`docs/README_INTEGRATIONS.md`](docs/README_INTEGRATIONS.md) - All integrations
- [`docs/CHRONICLE_QUICK_START.md`](docs/CHRONICLE_QUICK_START.md) - Chronicle setup
- [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md) - What changed & path updates

## 🔧 Path Updates

### Models are now in:
```
core/anomaly_outputs/
```

### Test data is now in:
```
tests/test_logs_*.json
```

### To use from other directories:
```python
# Option 1: Relative path
MODEL_DIR = "../core/anomaly_outputs"

# Option 2: Environment variable
import os
MODEL_DIR = os.getenv('MODEL_DIR', 'C:/Users/jimmy/.local/bin/anomaly-detection/core/anomaly_outputs')
```

## 🧪 Verify Everything Works

```bash
# 1. Test core detection
cd core/
python log_anomaly_detection_lite.py --data_path ../tests/test_logs_attack.json

# 2. Test API
cd ../api/
python anomaly_api.py &
python test_api.py
kill %1  # Stop API

# 3. Test batch
cd ../batch/
python batch_processor.py --log-dir ../tests/ --once
```

If all tests pass, you're ready to go! ✅

## 🆘 Troubleshooting

**"Module not found"**
- Install dependencies: `pip install -r config/requirements_minimal.txt`
- Check you're in the right directory

**"Models not found"**
- Train models first: `cd core && python log_anomaly_detection_lite.py --data_path ../tests/`
- Check path: models should be in `core/anomaly_outputs/`

**"Test data not found"**
- Test files are in: `tests/test_logs_*.json`
- Use relative path: `../tests/` from core/api/batch directories

**Paths don't work**
- See [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md) for path update guide
- Use absolute paths if needed
- Set environment variable `ANOMALY_HOME`

## 🌟 What's New?

✨ **Better Organization**
- Files grouped by function
- Each component has its own README
- Clearer project structure

✨ **Easier Navigation**
- Find files faster
- Clear separation of concerns
- Better for team collaboration

✨ **Production Ready**
- Docker deployment configured
- Multi-service architecture
- Scalable design

## 🎓 Next Steps

1. **Explore components** - Read README in each directory
2. **Try examples** - Run code in `examples/integration_examples.py`
3. **Choose deployment** - API, MCP, or Docker
4. **Connect to SIEM** - Chronicle, Splunk, or Elasticsearch
5. **Go to production** - Use Docker or cloud deployment

## 📖 Full Documentation

Browse all docs in [`docs/`](docs/) directory or start with:
- **README.md** (main project README)
- **docs/QUICK_START.md** (detailed setup)
- **docs/README_INTEGRATIONS.md** (all integration options)

---

**Ready to detect threats!** 🛡️

Quick commands:
```bash
cd core/          # Analyze logs
cd api/           # Start API
cd chronicle/     # Setup Chronicle
cd mcp/           # Setup Claude
```
