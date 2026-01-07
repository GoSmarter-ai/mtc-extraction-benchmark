#!/bin/bash
# Quick validation script for the devcontainer setup
# Run this after opening the devcontainer to verify everything is working
# Note: We intentionally don't use 'set -e' because we want to show all validation results

echo "====== DevContainer Validation ======"
echo ""

# Check Python version
echo "1. Python Version:"
python --version
echo ""

# Check key Python packages
echo "2. Checking Python packages..."
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "  ✗ PyTorch not found"
python -c "import transformers; print(f'  ✓ Transformers {transformers.__version__}')" 2>/dev/null || echo "  ✗ Transformers not found"
python -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__}')" 2>/dev/null || echo "  ✗ OpenCV not found"
python -c "import numpy; print(f'  ✓ NumPy {numpy.__version__}')" 2>/dev/null || echo "  ✗ NumPy not found"
python -c "import pandas; print(f'  ✓ Pandas {pandas.__version__}')" 2>/dev/null || echo "  ✗ Pandas not found"
echo ""

# Check OCR tools
echo "3. Checking OCR tools..."
tesseract --version 2>&1 | head -1 | sed 's/^/  ✓ /' || echo "  ✗ Tesseract not found"
python -c "from paddleocr import PaddleOCR; print('  ✓ PaddleOCR available')" 2>/dev/null || echo "  ✗ PaddleOCR not available"
python -c "from doctr.models import ocr_predictor; print('  ✓ DocTR available')" 2>/dev/null || echo "  ✗ DocTR not available"
echo ""

# Check GPU support
echo "4. GPU Support:"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null
echo ""

# Check development tools
echo "5. Development Tools:"
which jupyter >/dev/null 2>&1 && echo "  ✓ Jupyter" || echo "  ✗ Jupyter"
which black >/dev/null 2>&1 && echo "  ✓ Black" || echo "  ✗ Black"
which flake8 >/dev/null 2>&1 && echo "  ✓ Flake8" || echo "  ✗ Flake8"
which pytest >/dev/null 2>&1 && echo "  ✓ Pytest" || echo "  ✗ Pytest"
echo ""

# Check GitHub CLI
echo "6. GitHub Integration:"
which gh >/dev/null 2>&1 && echo "  ✓ GitHub CLI installed" || echo "  ✗ GitHub CLI not found"
[ -n "$GITHUB_TOKEN" ] && echo "  ✓ GITHUB_TOKEN is set" || echo "  ℹ GITHUB_TOKEN not set (needed for GitHub Models)"
echo ""

echo "====== Validation Complete ======"
echo ""
echo "Next steps:"
echo "  - Run 'jupyter lab' to start Jupyter"
echo "  - See docs/devcontainer-iteration.md for customization"
echo "  - See docs/github-models-integration.md for LLM integration"
