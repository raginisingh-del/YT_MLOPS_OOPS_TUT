#!/bin/bash
# Train model script

set -e

echo "========================================="
echo "MLOps Model Training Script"
echo "========================================="

# Setup environment
echo "Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run training
echo "Starting model training..."
python examples/pipeline_example.py

echo "========================================="
echo "Training completed successfully!"
echo "========================================="
