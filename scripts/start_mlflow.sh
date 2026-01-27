#!/bin/bash
# Start MLflow UI script

set -e

echo "========================================="
echo "Starting MLflow UI"
echo "========================================="

# Start MLflow UI
mlflow ui --backend-store-uri file:///$(pwd)/mlruns --host 0.0.0.0 --port 5000

echo "MLflow UI available at http://localhost:5000"
