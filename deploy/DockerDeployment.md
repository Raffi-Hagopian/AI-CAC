# VA-AI-CAC Docker/Podman Deployment Guide

## Overview

This guide documents the containerization of the VA-AI-CAC (Veterans Affairs AI-based Coronary Artery Calcium scoring) application for deployment with GPU acceleration using Docker/Podman.

## Requirements Deviation

### Original Specification (requirements.txt)
- CUDA 11.3
- PyTorch 1.12.1+cu113
- Torchvision 0.13.1+cu113

### Actual Implementation (Dockerfile.updated)
- CUDA 11.8
- PyTorch 2.0.1+cu118
- Torchvision 0.15.2

### Why We Deviated

The original requirements could not be containerized due to:

1. **PyTorch 1.12.1 wheels archived** - PyTorch removed old wheel distributions from their repository
2. **CUDA 11.3 base images unavailable** - NVIDIA doesn't provide CUDA 11.3 images for Ubuntu 22.04
3. **Ubuntu 20.04 PPA failures** - Deadsnakes PPA had connectivity/authentication issues
4. **CUDA version compatibility** - Initial builds used CUDA 13.0, incompatible with host driver (CUDA 12.0.2)
5. **MONAI dependency conflicts** - `monai[all]==1.4.0` automatically upgraded PyTorch to incompatible versions

### Working Solution

- **Base Image**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **Python**: 3.10 (native in Ubuntu 22.04, no PPA needed)
- **PyTorch**: 2.0.1+cu118
- **CUDA**: 11.8 (backward compatible with NVIDIA driver 535.129.03)
- **MONAI**: 1.4.0 (base package, without `[all]` to avoid dependency issues)

## Quick Start

### Build the Container

```bash
cd /path/to/VA-AI-CAC

# Clean previous builds
podman rm -f va-ai-cac >/dev/null
podman rmi -f localhost/va-ai-cac >/dev/null
podman system prune -a -f

# Build from Dockerfile
podman build --no-cache -f Dockerfile -t va-ai-cac:latest .
```

### Run with GPU Support

```bash
podman run -d --name va-ai-cac \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --security-opt=label=disable \
  -p 5000:25000 \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/model:/app/model \
  va-ai-cac:latest
```

### Verify GPU Access

```bash
# Check container logs
podman logs va-ai-cac | grep "Device:"
# Expected: Device: cuda:0

# Verify PyTorch version
podman exec va-ai-cac python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU Available: {torch.cuda.is_available()}')"
# Expected: PyTorch: 2.0.1+cu118, CUDA: 11.8, GPU Available: True

# Test GPU
podman exec va-ai-cac nvidia-smi
```

## Running Predictions

### Health Check

```bash
curl http://localhost:5000/health
```

### Submit CT Study

```bash
curl -X POST http://localhost:5000/predict \
  -F "study_zip=@/path/to/CT_Study.zip" \
  -F "save_masks=true"
```

## Key Configuration Details

### Dockerfile Highlights

```dockerfile
# CUDA 11.8 base image (compatible with driver 535+)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Pin setuptools to avoid deprecation warnings
RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel

# Install PyTorch with explicit CUDA 11.8 index
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchmetrics==1.5.2

# Install MONAI base (without [all] to prevent PyTorch upgrade)
RUN pip install --no-cache-dir "monai==1.4.0"

# Force reinstall PyTorch after MONAI dependencies
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1 \
    torchvision==0.15.2
```

### Host Requirements

- **NVIDIA Driver**: 535.129.03 or newer (CUDA 12.0.2+ support)
- **Podman**: 4.9+ or Docker 20.10+
- **NVIDIA Container Toolkit**: For GPU passthrough
- **OS**: Linux (tested on RHEL/CentOS)

## Performance

| Configuration | Processing Speed | Time (120-slice study) |
|---------------|------------------|------------------------|
| GPU (cuda:0) | 15-25 slices/sec | 5-8 seconds |
| CPU fallback | 2-3 slices/sec | 40-60 seconds |

## Troubleshooting

### Container Uses CPU Instead of GPU

```bash
# Verify GPU devices are accessible
podman exec va-ai-cac ls -la /dev/nvidia*

# Check CUDA availability
podman exec va-ai-cac python -c "import torch; print(torch.cuda.is_available())"

# If false, ensure devices are passed with --device flags
```

### Wrong PyTorch Version in Container

```bash
# Check version
podman exec va-ai-cac python -c "import torch; print(torch.__version__)"

# If incorrect, rebuild with complete cache clear
podman system prune -a -f
podman build --no-cache --pull -f Dockerfile -t va-ai-cac:latest .
```

### Port 5000 Already in Use

```bash
# Find process using the port
sudo lsof -i :5000

# Use different host port
podman run -d --name va-ai-cac -p 5001:25000 ... va-ai-cac:latest
```

### CUDA Initialization Error

If you see "CUDA initialization failed" or "driver too old" errors:

1. **Check host driver version**:
   ```bash
   nvidia-smi
   ```
   Should show driver 535+ with CUDA 12.0+

2. **Verify container CUDA version**:
   ```bash
   podman exec va-ai-cac nvcc --version
   ```
   Should show CUDA 11.8

3. **Ensure driver compatibility**: CUDA 11.8 requires driver 450.80.02+ (you have 535.129.03 ✓)

## Version Compatibility Matrix

| Component | requirements.txt | Actual Container | Status |
|-----------|------------------|------------------|--------|
| Python | 3.10 | 3.10 | ✅ Match |
| PyTorch | 1.12.1+cu113 | 2.0.1+cu118 | ⚠️ Upgraded (wheels unavailable) |
| torchvision | 0.13.1+cu113 | 0.15.2 | ⚠️ Upgraded |
| MONAI | 1.4.0 | 1.4.0 | ✅ Match |
| numpy | 1.26.4 | 1.26.4 | ✅ Match |
| scipy | 1.13.1 | 1.13.1 | ✅ Match |
| SimpleITK | 2.4.1 | 2.4.1 | ✅ Match |
| pydicom | 2.4.4 | 2.4.4 | ✅ Match |

## Container Management

```bash
# View running containers
podman ps

# Stop container
podman stop va-ai-cac

# Start stopped container
podman start va-ai-cac

# Restart container
podman restart va-ai-cac

# View logs (live)
podman logs -f va-ai-cac

# Remove container
podman rm -f va-ai-cac

# Execute commands inside container
podman exec -it va-ai-cac bash
```

## Storage Cleanup

```bash
# Check storage usage
du -sh storage/incoming
du -sh storage/outputs/masks

# Clean predictions older than 7 days
find storage/incoming -type d -mtime +7 -exec rm -rf {} +
find storage/outputs/masks -type d -mtime +7 -exec rm -rf {} +
```

## Running Multiple Instances

To run 2 instances of the container:

```bash
# Instance 1
podman run --rm -d \
  -p 5000:25000 \
  --shm-size=2g \
  --device nvidia.com/gpu=all \
  --env-file .env \
  -e FLASK_DEBUG=false \
  -v ${PWD}/storage:/app/storage \
  --name va-ai-cac-1 \
  va-ai-cac

# Instance 2
podman run --rm -d \
  -p 5001:25000 \
  --shm-size=2g \
  --device nvidia.com/gpu=all \
  --env-file .env \
  -e FLASK_DEBUG=false \
  -v ${PWD}/storage:/app/storage \
  --name va-ai-cac-2 \
  va-ai-cac
```

**Key points:**
- Different container names (`va-ai-cac-1`, `va-ai-cac-2`)
- Different host ports (`5000`, `5001`)
- Different storage volumes to avoid conflicts

## References

- **NVIDIA CUDA Compatibility**: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- **PyTorch CUDA Builds**: https://pytorch.org/get-started/locally/
- **MONAI Documentation**: https://docs.monai.io/
- **Podman GPU Access**: https://github.com/containers/podman/blob/main/docs/tutorials/podman-with-gpus.md
- **VA-AI-CAC Paper**: https://doi.org/10.1056/AIoa2400937

## Files

- `Dockerfile` - Working Dockerfile with CUDA 11.8 + PyTorch 2.0.1
- `requirements.Dockerfile.txt` - Python dependencies matching container build
- `LOCAL_SETUP.md` - Local development setup (without containers)

## Author

**Muazzam Khan**  
Deployment Date: 2026-06-30

## Model Attribution

**VA-AI-CAC Model**  
Creator: Raffi Hagopian, MD  
U.S. Department of Veterans Affairs  
Citation: Hagopian, R., et al. (2024). NEJM AI. DOI: 10.1056/AIoa2400937
