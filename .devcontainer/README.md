# DevContainer Configuration

This directory contains the development container configuration for the MTC Extraction Benchmark project.

## Files

- **`devcontainer.json`**: VS Code devcontainer configuration
  - Defines VS Code settings and extensions
  - Configures port forwarding for Jupyter (8888) and web services (5000)
  - Sets up the container user and permissions
  
- **`Dockerfile`**: Container image definition
  - Based on NVIDIA CUDA for GPU support
  - Python 3.11 with ML/AI packages
  - OCR tools (Tesseract, PaddleOCR, DocTR)
  - Development tools (Black, Flake8, Pytest)

- **`validate-setup.sh`**: Quick validation script
  - Run this after container starts to verify setup
  - Checks Python packages, OCR tools, and GPU availability

## Quick Start

### Using GitHub Codespaces

1. Go to the repository on GitHub
2. Click "Code" → "Codespaces" → "Create codespace"
3. Wait for container to build (first time only, ~5-10 minutes)
4. Run `bash .devcontainer/validate-setup.sh` to verify setup

### Using VS Code Locally

1. Install [Docker](https://www.docker.com/products/docker-desktop)
2. Install [VS Code](https://code.visualstudio.com/)
3. Install [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
4. Open repository in VS Code
5. Click "Reopen in Container" when prompted (or use Command Palette → "Dev Containers: Reopen in Container")
6. Wait for build to complete
7. Run `bash .devcontainer/validate-setup.sh` to verify setup

## Container Specifications

**Base Image:** `nvidia/cuda:12.2.0-runtime-ubuntu22.04`

**Python Version:** 3.11

**Key Packages:**
- PyTorch with CUDA 12.1 support
- Transformers, Datasets, Accelerate
- OpenCV, Pillow, NumPy, Pandas
- Tesseract, PaddleOCR, DocTR
- Jupyter Lab
- Black, Flake8, Pytest

**System Requirements:**
- Minimum 4 GB RAM (8 GB recommended)
- 20 GB disk space
- GPU optional but recommended for model training

## Customization

To customize the devcontainer:

1. Edit `devcontainer.json` for VS Code settings and extensions
2. Edit `Dockerfile` to add/remove packages
3. Rebuild container: Command Palette → "Dev Containers: Rebuild Container"

See the [DevContainer Iteration Guide](../docs/devcontainer-iteration.md) for detailed instructions.

## Troubleshooting

### Container build fails

- **Out of disk space**: Run `docker system prune -a`
- **Network issues**: Check internet connection and Docker registry access
- **Permission errors**: Ensure Docker has proper permissions

### Packages not found

- Run: Command Palette → "Dev Containers: Rebuild Container Without Cache"
- This forces a clean rebuild

### GPU not detected

- Ensure NVIDIA drivers are installed on host
- Check Docker has GPU access: `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`
- GPU support requires NVIDIA GPU with compatible drivers

### Slow performance

- Increase Docker memory allocation (Docker Settings → Resources)
- Use GPU-enabled Codespace machine type (8+ cores)
- Close unnecessary applications

## Further Reading

- [VS Code DevContainer Docs](https://code.visualstudio.com/docs/devcontainers/containers)
- [DevContainer Specification](https://containers.dev/)
- [GitHub Codespaces Docs](https://docs.github.com/en/codespaces)
- [Project Documentation](../docs/)
