#!/bin/bash
# Wrapper script to run PPO training with proper environment activation

set -e

VENV_NAME="clause_ppo_venv"

# Check if virtual environment exists
if [ ! -d "$VENV_NAME" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   ./setup_venv.sh"
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment: $VENV_NAME"
source $VENV_NAME/bin/activate

# Verify PyTorch is available
echo "🔍 Verifying PyTorch installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} ready')" || {
    echo "❌ PyTorch not found. Please run setup_venv.sh first"
    exit 1
}

# Run PPO training
echo "🚀 Starting PPO training..."
./scripts/run_ppo_training.sh