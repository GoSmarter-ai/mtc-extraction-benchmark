# Quick Start Guide

Get up and running with the MTC Extraction Benchmark project in minutes!

## Option 1: GitHub Codespaces (Recommended for Quick Start)

**Fastest way to get started - no local setup required!**

1. **Create a Codespace:**
   - Visit: https://github.com/GoSmarter-ai/mtc-extraction-benchmark
   - Click the green "**Code**" button
   - Select "**Codespaces**" tab
   - Click "**Create codespace on main**"

2. **Wait for Setup:**
   - First time: ~5-10 minutes (building container)
   - Subsequent times: ~30 seconds (using cached image)
   - Watch the terminal for progress

3. **Validate Setup:**
   ```bash
   bash .devcontainer/validate-setup.sh
   ```

4. **Start Exploring:**
   ```bash
   # Start Jupyter Lab
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
   
   # Or create your first Python script
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

5. **Access Jupyter:**
   - Click the "Ports" tab in VS Code
   - Click the link for port 8888
   - Or click the popup notification

**Cost:** Free for 120 core-hours/month (personal account) or 180 core-hours/month (Pro)

## Option 2: VS Code with Docker (Local Development)

**Best for offline work and full control over resources.**

### Prerequisites

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Install [VS Code](https://code.visualstudio.com/)
3. Install [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/GoSmarter-ai/mtc-extraction-benchmark.git
   cd mtc-extraction-benchmark
   ```

2. **Open in VS Code:**
   ```bash
   code .
   ```

3. **Open in Container:**
   - VS Code will detect the devcontainer configuration
   - Click "**Reopen in Container**" in the popup
   - Or: Command Palette (F1) → "**Dev Containers: Reopen in Container**"

4. **Wait for Build:**
   - First time: ~10-15 minutes (downloads and installs everything)
   - Status shown in bottom-right corner
   - Terminal shows build progress

5. **Validate Setup:**
   ```bash
   bash .devcontainer/validate-setup.sh
   ```

6. **Start Working:**
   ```bash
   # Your terminal is now inside the container with all tools ready!
   python --version  # Should show Python 3.11.x
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
   ```

## What's Included?

The devcontainer provides a complete environment with:

✅ **Python 3.11** with pip, virtualenv  
✅ **ML/AI Frameworks**: PyTorch (with CUDA), Transformers, Datasets  
✅ **OCR Tools**: Tesseract, PaddleOCR, DocTR, Docling  
✅ **Data Processing**: NumPy, Pandas, OpenCV, Pillow  
✅ **PDF Tools**: PyPDF2, pdfplumber, PyMuPDF, pdf2image  
✅ **Development**: Jupyter Lab, Black, Flake8, Pytest  
✅ **Version Control**: Git, GitHub CLI  
✅ **VS Code Extensions**: Python, Jupyter, Copilot, and more

## Next Steps

### 1. Explore the Documentation

- **[DevContainer Iteration](./docs/devcontainer-iteration.md)**: Customize your environment
- **[Codespaces Guide](./docs/codespaces-guide.md)**: Advanced Codespaces features
- **[GitHub Models](./docs/github-models-integration.md)**: Use LLMs for extraction

### 2. Set Up Your Workspace

```bash
# Create directory structure
mkdir -p data/{raw,processed,ground_truth}
mkdir -p src/{ocr,extraction,evaluation}
mkdir -p notebooks
mkdir -p scripts
mkdir -p tests
```

### 3. Install Additional Packages (if needed)

```bash
# Add to requirements.txt
echo "your-package-name" >> requirements.txt
pip install -r requirements.txt
```

### 4. Test OCR Tools

```python
# Test Tesseract
from PIL import Image
import pytesseract

# Create a simple test (or use your own image)
print("Tesseract version:", pytesseract.get_tesseract_version())

# Test PaddleOCR
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
print("PaddleOCR initialized successfully!")

# Test DocTR
from doctr.models import ocr_predictor
model = ocr_predictor(pretrained=True)
print("DocTR model loaded successfully!")
```

### 5. Set Up GitHub Models (Optional)

For LLM-based extraction:

1. **Get GitHub Token:**
   - Go to https://github.com/settings/tokens
   - Create token with `read:user` scope
   
2. **Set Environment Variable:**
   ```bash
   # In Codespaces (automatic)
   echo $GITHUB_TOKEN  # Should be set automatically
   
   # Locally, add to .env file
   echo "GITHUB_TOKEN=your_token_here" >> .env
   ```

3. **Test GitHub Models:**
   ```python
   import os
   from openai import OpenAI
   
   client = OpenAI(
       base_url="https://models.inference.ai.azure.com",
       api_key=os.environ["GITHUB_TOKEN"],
   )
   
   response = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[{"role": "user", "content": "Hello!"}]
   )
   print(response.choices[0].message.content)
   ```

## Common Commands

```bash
# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Run Python script
python scripts/your_script.py

# Run tests
pytest tests/

# Format code
black src/

# Lint code
flake8 src/

# Check git status
git status

# Push changes (Codespaces)
git add .
git commit -m "Your message"
git push
```

## Troubleshooting

### Container won't build
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
# In VS Code: F1 → "Dev Containers: Rebuild Container Without Cache"
```

### Package not found
```bash
# Upgrade pip
pip install --upgrade pip

# Reinstall package
pip install --force-reinstall package-name
```

### Jupyter won't start
```bash
# Check if port is already in use
lsof -i :8888

# Try different port
jupyter lab --ip=0.0.0.0 --port=8889 --no-browser
```

### GPU not detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Note: GPU support requires:
# - NVIDIA GPU on host machine
# - NVIDIA drivers installed
# - Docker configured for GPU access
```

## Getting Help

- **Documentation**: Check the `docs/` folder
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **VS Code**: F1 → "Help: Report Issue"

## Tips

💡 **Use Codespaces for experimentation** - Free tier is generous  
💡 **Use local Docker for intensive training** - Better for large models  
💡 **Enable Settings Sync** - Keep VS Code settings consistent  
💡 **Use Git often** - Commit and push regularly  
💡 **Stop Codespaces when done** - Save on core-hours  

---

**Ready to start extracting MTC data? Happy coding! 🚀**
