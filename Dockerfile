FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Install MLflow for experiment tracking
RUN pip install mlflow

# Create working directory
WORKDIR /workspace

# Copy source code
COPY . .

# Default command
CMD ["python", "run_all_kaggle.py", "--results-dir", "/workspace/results"]
