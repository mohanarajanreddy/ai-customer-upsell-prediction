#!/bin/bash
echo "🚀 AI Customer Upsell Prediction System Setup"
echo "Repository: github.com/mohanarajanreddy/ai-customer-upsell-prediction"
echo "Author: Mohanarajan P"
echo ""

# Check GPU
echo "🎮 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✅ NVIDIA GPU detected"
    USE_GPU=true
else
    echo "⚠️  No GPU detected - using CPU mode"
    USE_GPU=false
fi

# Update system
echo "📦 Updating system..."
sudo apt update -qq && sudo apt upgrade -y -qq

# Install Python
echo "🐍 Installing Python 3.10..."
sudo apt install -y python3.10 python3.10-venv python3-pip build-essential

# Create virtual environment
echo "🌐 Creating virtual environment..."
python3.10 -m venv gpu_venv
source gpu_venv/bin/activate

# Install packages
echo "⚡ Installing packages..."
pip install --upgrade pip setuptools wheel

if [ "$USE_GPU" = true ]; then
    pip install -r requirements-gpu.txt
    pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11 cuml-cu11 2>/dev/null || echo "⚠️  RAPIDS optional"
else
    pip install pandas scikit-learn streamlit plotly seaborn matplotlib numpy scipy joblib tqdm python-dotenv jupyter pytest
fi

# Set permissions
chmod +x scripts/*.sh

echo ""
echo "🎉 Setup Complete!"
echo "Next: source gpu_venv/bin/activate && streamlit run src/dashboard/app.py"
