#!/bin/bash
# Run tests script

set -e

echo "========================================="
echo "MLOps Testing Script"
echo "========================================="

# Setup environment
echo "Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run tests
echo "Running tests..."
pytest tests/ -v --cov=src/mlops --cov-report=term-missing

echo "========================================="
echo "Tests completed successfully!"
echo "========================================="
