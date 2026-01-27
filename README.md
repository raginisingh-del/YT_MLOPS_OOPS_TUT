# MLOps Pipeline - Machine Learning Operations

![CI/CD](https://github.com/raginisingh-del/YT_MLOPS_OOPS_TUT/workflows/MLOps%20CI/CD%20Pipeline/badge.svg)

A comprehensive MLOps (Machine Learning Operations) project that demonstrates building scalable pipelines for data processing, model training, versioning, deployment, and monitoring. This project focuses on reproducibility, automation, and continuous improvement of models in production.

## 🎯 Overview

MLOps integrates Machine Learning, DevOps, and Data Engineering to automate and manage the complete ML lifecycle. This project delivers reliable and efficient machine learning systems through:

- **Data Processing**: Scalable pipelines for data cleaning, transformation, and feature engineering
- **Model Training**: Automated training with hyperparameter management
- **Model Versioning**: MLflow integration for experiment tracking and model versioning
- **Deployment**: Model registry and serving infrastructure
- **Monitoring**: Performance tracking, data drift detection, and alerting

## 📁 Project Structure

```
YT_MLOPS_OOPS_TUT/
├── src/mlops/              # Main package
│   ├── data/               # Data processing modules
│   │   └── pipeline.py     # Data pipelines, cleaning, splitting
│   ├── models/             # Model training modules
│   │   └── trainer.py      # Model training, versioning
│   ├── deployment/         # Deployment modules
│   │   └── server.py       # Model registry, serving
│   ├── monitoring/         # Monitoring modules
│   │   └── metrics.py      # Performance tracking, drift detection
│   └── utils/              # Utility modules
│       ├── config.py       # Configuration management
│       └── logging.py      # Logging setup
├── config/                 # Configuration files
│   └── config.yaml         # Main configuration
├── examples/               # Example scripts
│   └── pipeline_example.py # End-to-end pipeline example
├── tests/                  # Test suite
│   ├── test_data_pipeline.py
│   └── test_models.py
├── data/                   # Data directories
│   ├── raw/                # Raw data
│   └── processed/          # Processed data
├── models/                 # Saved models
├── logs/                   # Log files
├── .github/workflows/      # CI/CD workflows
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
└── setup.py                # Package setup
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/raginisingh-del/YT_MLOPS_OOPS_TUT.git
cd YT_MLOPS_OOPS_TUT
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

### Running the Pipeline

Run the complete MLOps pipeline example:

```bash
python examples/pipeline_example.py
```

This will execute:
1. Data generation and processing
2. Model training with versioning
3. Model deployment
4. Monitoring and performance tracking

## 📚 Core Components

### 1. Data Processing Pipeline

The data processing module provides:

- **DataCleaner**: Handle missing values and outliers
- **FeatureEngineer**: Create and transform features
- **DataSplitter**: Split data into train/val/test sets
- **DataScaler**: Normalize and standardize features
- **DataPipeline**: Chain multiple processors together

Example:
```python
from mlops.data.pipeline import DataCleaner, DataPipeline, DataSplitter

pipeline = DataPipeline([
    DataCleaner(fill_strategy="mean"),
])
processed_data = pipeline.run(raw_data)

splitter = DataSplitter(test_size=0.2, val_size=0.1)
X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)
```

### 2. Model Training & Versioning

The models module supports:

- **ClassificationModel**: Classification tasks (Random Forest, Logistic Regression)
- **RegressionModel**: Regression tasks (Random Forest, Linear Regression)
- **ModelVersioning**: MLflow integration for experiment tracking
- **ModelTrainer**: Orchestrate training pipeline

Example:
```python
from mlops.models.trainer import ClassificationModel, ModelTrainer, ModelVersioning

model = ClassificationModel(
    model_name="my_classifier",
    model_type="random_forest",
    n_estimators=100,
    max_depth=10
)

versioning = ModelVersioning(experiment_name="my_experiment")
trainer = ModelTrainer(model, versioning)
metrics = trainer.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
```

### 3. Model Deployment

The deployment module provides:

- **ModelRegistry**: Manage model versions
- **ModelServer**: Serve models for inference
- **PredictionService**: High-level prediction API
- **DeploymentManager**: Handle deployment lifecycle

Example:
```python
from mlops.deployment.server import ModelRegistry, PredictionService, DeploymentManager

registry = ModelRegistry(registry_path="models")
registry.register_model("my_model", model_path, version="1.0")

service = PredictionService(registry)
service.load_model("my_model", version="1.0")
predictions = service.predict("my_model", data)
```

### 4. Monitoring & Alerting

The monitoring module includes:

- **PerformanceMonitor**: Track model performance over time
- **DataDriftDetector**: Detect distribution shifts in production data
- **AlertManager**: Manage threshold-based alerts
- **ModelMonitor**: Comprehensive monitoring system

Example:
```python
from mlops.monitoring.metrics import ModelMonitor

monitor = ModelMonitor(
    model_name="my_model",
    reference_data=X_train,
    alert_config={"accuracy_threshold": 0.7}
)

monitor.monitor_prediction(X_new, y_true, y_pred)
report = monitor.get_monitoring_report()
```

## 🔧 Configuration

Configure the pipeline using `config/config.yaml`:

```yaml
data:
  test_size: 0.2
  val_size: 0.1
  random_state: 42

training:
  model_type: "random_forest"
  hyperparameters:
    n_estimators: 100
    max_depth: 10

monitoring:
  drift_threshold: 0.1
  alert_thresholds:
    accuracy_threshold: 0.7
```

## 🧪 Testing

Run tests with pytest:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/mlops --cov-report=term-missing

# Run specific test file
pytest tests/test_data_pipeline.py -v
```

## 🔄 CI/CD Pipeline

The project includes GitHub Actions workflows for:

- **Continuous Integration**: Automated testing on multiple Python versions
- **Code Quality**: Linting with flake8
- **Build**: Package building and artifact generation

Workflow runs automatically on pushes to `main` and `develop` branches.

## 📊 MLflow Integration

Track experiments with MLflow:

```bash
# Start MLflow UI
mlflow ui

# View at http://localhost:5000
```

The pipeline automatically logs:
- Hyperparameters
- Metrics (accuracy, precision, recall, F1)
- Model artifacts
- Run metadata

## 🛠️ Development

### Adding New Features

1. Create your feature in the appropriate module
2. Add tests in the `tests/` directory
3. Update documentation
4. Run tests and ensure they pass
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Document classes and functions with docstrings
- Keep functions focused and single-purpose

## 📈 Best Practices

This project demonstrates MLOps best practices:

1. **Reproducibility**: Fixed random seeds, version control, configuration management
2. **Automation**: CI/CD pipelines, automated testing, automated deployment
3. **Monitoring**: Performance tracking, data drift detection, alerting
4. **Scalability**: Modular design, pipeline abstractions, containerization-ready
5. **Maintainability**: Clean code, comprehensive tests, documentation

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MLflow for experiment tracking
- scikit-learn for ML algorithms
- pytest for testing framework

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project demonstrating MLOps concepts and best practices. For production use, additional considerations like security, scalability, and infrastructure management should be addressed.
