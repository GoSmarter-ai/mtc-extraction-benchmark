# Iterating on the DevContainer

This guide explains how to modify and test changes to the devcontainer configuration for the MTC Extraction Benchmark project.

## Overview

The devcontainer is configured in the `.devcontainer` directory and consists of:
- `devcontainer.json`: Configuration for VS Code and container settings
- `Dockerfile`: The container image definition with all dependencies

## Making Changes to the DevContainer

### 1. Modifying the Dockerfile

The Dockerfile contains all the system packages and Python dependencies needed for the project.

**Common modifications:**

#### Adding a new Python package
```dockerfile
# Add to the appropriate RUN pip install section
RUN pip install --no-cache-dir \
    your-new-package \
    another-package
```

#### Adding a system package
```dockerfile
# Add to the appropriate RUN apt-get install section
RUN apt-get update && apt-get install -y \
    your-system-package \
    && rm -rf /var/lib/apt/lists/*
```

#### Updating Python version
```dockerfile
# Change the ARG at the top of the Dockerfile
ARG PYTHON_VERSION=3.12  # Update version here
```

### 2. Modifying devcontainer.json

The `devcontainer.json` file configures VS Code settings and container behavior.

**Common modifications:**

#### Adding VS Code extensions
```json
"extensions": [
    "ms-python.python",
    "your-new-extension-id"
]
```

#### Changing Python settings
```json
"settings": {
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true  // Enable pylint
}
```

#### Adding port forwarding
```json
"forwardPorts": [8888, 5000, 3000],  // Add new port
```

#### Modifying post-create commands
```json
"postCreateCommand": "pip install -r requirements.txt && pre-commit install"
```

## Rebuilding the Container

### Method 1: Full Rebuild (VS Code)

1. Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P)
2. Type "Dev Containers: Rebuild Container"
3. Select "Dev Containers: Rebuild Container Without Cache" for a clean rebuild

**When to use:** After modifying the Dockerfile or when you want to ensure a clean build.

### Method 2: Reload Window (VS Code)

1. Open the Command Palette
2. Type "Developer: Reload Window"

**When to use:** After only modifying `devcontainer.json` settings (not Dockerfile).

### Method 3: Command Line Rebuild

From outside the container:
```bash
# Rebuild the container using Docker
docker compose -f .devcontainer/docker-compose.yml build --no-cache

# Or rebuild and start
docker compose -f .devcontainer/docker-compose.yml up --build
```

## Testing Your Changes

### 1. Verify Python Environment

After rebuilding, open a terminal in VS Code and verify:

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Test specific package
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

### 2. Test OCR Tools

```bash
# Test Tesseract
tesseract --version

# Test PaddleOCR (in Python)
python -c "from paddleocr import PaddleOCR; print('PaddleOCR imported successfully')"

# Test DocTR
python -c "from doctr.models import ocr_predictor; print('DocTR imported successfully')"
```

### 3. Test GPU Support (if available)

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### 4. Test Jupyter

```bash
# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Then access via the forwarded port in your browser.

## Common Issues and Solutions

### Issue: Build fails with "no space left on device"

**Solution:** Clean up Docker images and containers:
```bash
docker system prune -a
docker volume prune
```

### Issue: Python package installation fails

**Solution:** 
1. Check if the package name is correct on PyPI
2. Try installing with specific version: `package==1.2.3`
3. Check for compatibility issues with Python version

### Issue: Extensions not installing

**Solution:**
1. Check extension ID is correct
2. Ensure the extension is compatible with the VS Code version
3. Try rebuilding without cache

### Issue: GPU not detected

**Solution:**
1. Ensure NVIDIA drivers are installed on host
2. Verify Docker has GPU access: `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`
3. Update `devcontainer.json` to include GPU access:
```json
"runArgs": ["--gpus", "all", "--shm-size=2gb"]
```

## Best Practices

1. **Layer optimization**: Group related RUN commands to reduce image layers
2. **Cache usage**: Order Dockerfile commands from least to most frequently changed
3. **Clean up**: Always remove apt cache with `rm -rf /var/lib/apt/lists/*`
4. **Pin versions**: For production, pin specific versions of critical packages
5. **Document changes**: Update this guide when making significant changes

## Advanced: Using Docker Compose

For more complex setups, you can create a `docker-compose.yml` in `.devcontainer/`:

```yaml
version: '3.8'

services:
  app:
    build:
      context: ..
      dockerfile: .devcontainer/Dockerfile
    volumes:
      - ..:/workspace:cached
    command: sleep infinity
    environment:
      - PYTHONPATH=/workspace
    shm_size: '2gb'
```

Then reference it in `devcontainer.json`:
```json
{
  "name": "MTC Extraction Benchmark",
  "dockerComposeFile": "docker-compose.yml",
  "service": "app",
  "workspaceFolder": "/workspace"
}
```

## Getting Help

- [VS Code DevContainer documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [DevContainer specification](https://containers.dev/)
- [Dockerfile reference](https://docs.docker.com/engine/reference/builder/)

## Next Steps

- See [Codespaces Guide](./codespaces-guide.md) for working with GitHub Codespaces
- See [GitHub Models Integration](./github-models-integration.md) for using GitHub Models API
