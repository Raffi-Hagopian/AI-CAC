# VA-AI-CAC Local Development Setup

## Prerequisites

### Required Software
- **Python 3.10** (exactly 3.10, not 3.11 or newer)
- **CUDA Toolkit 11.8** (if using GPU)
- **NVIDIA GPU Driver 535.129.03+** (for GPU support)

### Check Your System

**Windows (PowerShell):**
```powershell
# Check Python version
python --version  # Should show Python 3.10.x

# Check NVIDIA GPU
nvidia-smi
```

**Linux:**
```bash
# Check Python version
python3 --version  # Should show Python 3.10.x

# Check NVIDIA GPU
nvidia-smi
```

---

## Quick Setup (Automated)

### Windows with GPU (CUDA 11.8+)
```powershell
.\scripts\setup_local_dev.ps1
```

### Windows CPU Only (No GPU)
```powershell
# Faster setup, skips CUDA dependencies
.\scripts\setup_windows_cpu.ps1
```

### Linux/Mac
```bash
bash scripts/setup_local_dev.sh
```

---

## Manual Setup

### 1. Install Python 3.10

**Windows:**
- **Direct download:** [Python 3.10.11 (64-bit)](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)
- **Important:** Check "Add Python to PATH" during installation
- **Installation issues?** See [PYTHON_INSTALL_WINDOWS.md](PYTHON_INSTALL_WINDOWS.md) for detailed troubleshooting

**Linux (RHEL/CentOS):**
```bash
sudo yum install python3.10 python3.10-venv python3.10-devel
```

**Linux (Ubuntu):**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

### 2. Create Virtual Environment

**Navigate to project directory:**
```bash
cd /path/to/VA-AI-CAC
```

**Create venv:**
```bash
# Windows
python -m venv .venv

# Linux/Mac
python3.10 -m venv .venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

You should see `(.venv)` in your prompt.

### 4. Upgrade pip and Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip "setuptools<81" wheel

# Install PyTorch for CUDA 11.8 (GPU)
pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchmetrics==1.5.2

# Or for CPU only (no GPU):
pip install torch==2.0.1 torchvision==0.15.2 torchmetrics==1.5.2

# Install other dependencies
pip install -r requirements.updated.txt
```

### 5. Download Model Weights

**Windows:**
```powershell
.\scripts\download_model.ps1
```

**Linux/Mac:**
```bash
bash scripts/download_model.sh
```

**Or manually:**
```bash
mkdir -p model
curl -L -o model/va_non_gated_ai_cac_model.pth \
  https://github.com/Raffi-Hagopian/AI-CAC/releases/download/v1.0.0/va_non_gated_ai_cac_model.pth
```

### 6. Create Storage Directories

```bash
mkdir -p storage/incoming
mkdir -p storage/outputs/masks
mkdir -p storage/outputs/debug
```

### 7. Run the Application

```bash
# Activate venv first (if not already activated)
# Windows: .\.venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate

# Run Flask app
python app.py
```

You should see:
```
Device: cuda:0  (or cpu if no GPU)
AI-CAC model loaded successfully.
Starting AI-CAC Flask server...
 * Running on http://127.0.0.1:25000
```

### 8. Test the Application

**In another terminal:**
```bash
# Health check
curl http://localhost:25000/health

# Run prediction
curl -X POST http://localhost:25000/predict \
  -F "study_zip=@/path/to/CT_Study.zip" \
  -F "save_masks=true"
```

---

## Troubleshooting

### Python 3.10 Not Found

**Windows:**
- Make sure Python 3.10 is installed
- Add to PATH: `C:\Python310\` and `C:\Python310\Scripts\`

**Linux:**
```bash
# Check available Python versions
ls /usr/bin/python*

# Create symlink if needed
sudo ln -s /usr/bin/python3.10 /usr/local/bin/python
```

### CUDA Not Found (GPU)

If you see "Device: cpu" instead of "Device: cuda:0":

```python
# Test CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch:', torch.__version__)"
```

**If False:**
- Install NVIDIA GPU drivers (535.129.03+)
- Install CUDA Toolkit 11.8
- Reinstall PyTorch with CUDA support

### Import Errors

```bash
# Verify all packages installed
pip list | grep -E "torch|monai|numpy|pandas"

# Reinstall if needed
pip install --force-reinstall -r requirements.updated.txt
```

### Port 25000 Already in Use

```bash
# Find what's using the port
# Windows:
netstat -ano | findstr :25000

# Linux:
lsof -i :25000

# Kill the process or change port in app.py
```

---

## Development Workflow

### Activate Environment
```bash
# Every time you start working
# Windows: .\.venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
```

### Run Application
```bash
python app.py
```

### Run Tests (if available)
```bash
python -m pytest tests/
```

### Deactivate Environment
```bash
deactivate
```

---

## Environment Variables

You can customize behavior with environment variables:

```bash
# Windows (PowerShell)
$env:MODEL_CHECKPOINT_FILE = "model/va_non_gated_ai_cac_model.pth"
$env:INFERENCE_BATCH_SIZE = "8"
python app.py

# Linux/Mac
export MODEL_CHECKPOINT_FILE="model/va_non_gated_ai_cac_model.pth"
export INFERENCE_BATCH_SIZE="8"
python app.py
```

---

## Comparison: Local vs Container

| Feature | Local Development | Container (Podman) |
|---------|------------------|-------------------|
| Setup time | 10-15 minutes | 20-30 minutes (first build) |
| Disk space | ~5 GB | ~15 GB |
| Isolation | None (uses system Python) | Complete isolation |
| GPU support | Requires local CUDA | Easier GPU passthrough |
| Hot reload | Easy (just restart app.py) | Rebuild container |
| Debugging | Direct IDE integration | Requires exec into container |
| Production | Not recommended | Recommended |

**Use local development for:**
- Quick testing and debugging
- Development/experimentation
- Iterating on code changes

**Use containers for:**
- Production deployment
- Consistent environments
- GPU compatibility issues
- Team collaboration

---

## VS Code Integration

### 1. Install Python Extension
- Install "Python" extension by Microsoft

### 2. Select Interpreter
1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type "Python: Select Interpreter"
3. Choose `.venv` interpreter

### 3. Create launch.json

`.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Flask App",
            "type": "python",
            "request": "launch",
            "module": "flask",
            "env": {
                "FLASK_APP": "app.py",
                "FLASK_ENV": "development"
            },
            "args": [
                "run",
                "--host=0.0.0.0",
                "--port=25000",
                "--no-debugger",
                "--no-reload"
            ],
            "jinja": true,
            "justMyCode": false
        }
    ]
}
```

### 4. Debug
- Press `F5` to start debugging
- Set breakpoints in code
- Use debug console

---

## Next Steps

After setup:
1. ✅ Verify GPU: `curl http://localhost:25000/health` should show `"device": "cuda:0"`
2. ✅ Test prediction: `python test_prediction.py /path/to/CT_Study.zip`
3. ✅ Read the main [README-khan.MD](README-khan.MD) for usage examples

For production deployment, use the containerized version with Podman/Docker.
