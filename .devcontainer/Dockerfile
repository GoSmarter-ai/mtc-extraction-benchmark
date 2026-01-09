# Use Python 3.11 as base image with CUDA support for GPU acceleration
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Set Python version
ARG PYTHON_VERSION=3.11

# Install base dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*
    
# Import GPG keys for repositories
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776

# Import GPG keys for repositories
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776

# Add deadsnakes PPA for Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python and essential tools
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Install OCR dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies for computer vision and ML
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install common Python packages
RUN python -m pip install --upgrade pip setuptools wheel

# Install core Python ML/AI packages
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    pillow \
    opencv-python-headless

# Install PyTorch with CUDA support (using cu121 for broader compatibility)
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install Transformers and related HuggingFace libraries
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    sentencepiece \
    protobuf

# Install OCR and Document AI libraries
RUN pip install --no-cache-dir \
    pytesseract \
    python-doctr[torch] \
    paddlepaddle \
    paddleocr \
    pdf2image \
    PyPDF2 \
    pdfplumber \
    pymupdf

# Install Jupyter and related tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    ipykernel

# Install development tools
RUN pip install --no-cache-dir \
    black \
    flake8 \
    pytest \
    pytest-cov \
    mypy \
    ruff

# Install additional utilities
RUN pip install --no-cache-dir \
    tqdm \
    python-dotenv \
    requests \
    pyyaml \
    jsonschema

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Switch to non-root user
USER $USERNAME

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Set environment variables for better Python experience
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
