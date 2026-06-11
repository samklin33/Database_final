#!/usr/bin/env python3
"""
Simplified PRM training script that works with the current environment.
"""

import sys
import os
sys.path.insert(0, 'src')

# Test imports
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError:
    print("❌ PyTorch not available")
    sys.exit(1)

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError:
    print("❌ Transformers not available")
    sys.exit(1)

try:
    import yaml
    print(f"✅ PyYAML available")
except ImportError:
    print("❌ PyYAML not available - installing...")
    os.system("pip install pyyaml")
    import yaml

# Now try the actual training
from training.train_prm import train_prm

# Load config
with open('configs/prm_config.yaml') as f:
    config = yaml.safe_load(f)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# Run training
train_prm(
    config=config,
    spider_dir='data/spider',
    processed_dir='data/processed',
)