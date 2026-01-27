"""End-to-end MLOps pipeline example."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from mlops.data.pipeline import (
    DataCleaner, DataLoader, DataPipeline, DataScaler, DataSplitter
)
from mlops.deployment.server import DeploymentManager, ModelRegistry, PredictionService
from mlops.models.trainer import ClassificationModel, ModelTrainer, ModelVersioning
from mlops.monitoring.metrics import ModelMonitor
from mlops.utils.config import ConfigManager
from mlops.utils.logging import setup_logging


def generate_sample_data(n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
    """
    Generate sample classification data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        
    Returns:
        DataFrame with features and target
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df


def main():
    """Run the complete MLOps pipeline."""
    # Load configuration
    config = ConfigManager(Path("config/config.yaml"))
    
    # Setup logging
    log_level = getattr(logging, config.get("logging.level", "INFO"))
    setup_logging(
        log_file=Path(config.get("logging.log_file", "logs/mlops.log")),
        level=log_level
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Starting MLOps Pipeline")
    logger.info("=" * 80)
    
    # 1. Data Processing
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Data Processing")
    logger.info("=" * 80)
    
    # Generate sample data
    logger.info("Generating sample data...")
    df = generate_sample_data(n_samples=1000, n_features=20)
    
    # Save raw data
    raw_data_path = Path(config.get("data.raw_data_path", "data/raw"))
    raw_data_path.mkdir(parents=True, exist_ok=True)
    DataLoader.save_csv(df, raw_data_path / "sample_data.csv", index=False)
    
    # Create data pipeline
    pipeline = DataPipeline([
        DataCleaner(fill_strategy="mean"),
    ])
    
    # Process data
    df_processed = pipeline.run(df)
    
    # Save processed data
    processed_data_path = Path(config.get("data.processed_data_path", "data/processed"))
    processed_data_path.mkdir(parents=True, exist_ok=True)
    DataLoader.save_csv(df_processed, processed_data_path / "processed_data.csv", index=False)
    
    # Split features and target
    X = df_processed.drop('target', axis=1)
    y = df_processed['target']
    
    # Split data
    splitter = DataSplitter(
        test_size=config.get("data.test_size", 0.2),
        val_size=config.get("data.val_size", 0.1),
        random_state=config.get("data.random_state", 42)
    )
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)
    
    # Scale features
    scaler = DataScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Model Training
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Model Training")
    logger.info("=" * 80)
    
    # Create model
    model = ClassificationModel(
        model_name="sample_classifier",
        model_type=config.get("training.model_type", "random_forest"),
        **config.get("training.hyperparameters", {})
    )
    
    # Setup versioning
    versioning = ModelVersioning(
        experiment_name=config.get("training.experiment_name", "ml_pipeline"),
        tracking_uri=config.get("versioning.tracking_uri")
    )
    
    # Train model
    trainer = ModelTrainer(model, versioning)
    test_metrics = trainer.train_and_evaluate(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test
    )
    
    logger.info(f"Final test metrics: {test_metrics}")
    
    # Save model
    model_path = Path(config.get("deployment.model_registry_path", "models"))
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(model_path / "sample_classifier_v1.pkl")
    
    # 3. Model Deployment
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Model Deployment")
    logger.info("=" * 80)
    
    # Setup model registry
    registry = ModelRegistry(registry_path=model_path)
    registry.register_model(
        model_name="sample_classifier",
        model_path=model_path / "sample_classifier_v1.pkl",
        version="1.0",
        metadata={"metrics": test_metrics}
    )
    
    # Deploy model
    deployment_manager = DeploymentManager(registry)
    deployment_manager.deploy(
        model_name="sample_classifier",
        version="1.0",
        environment=config.get("deployment.default_environment", "production")
    )
    
    # Setup prediction service
    prediction_service = PredictionService(registry)
    prediction_service.load_model("sample_classifier", version="1.0")
    
    # Make predictions
    sample_predictions = prediction_service.predict("sample_classifier", X_test_scaled[:10])
    logger.info(f"Sample predictions: {sample_predictions}")
    
    # 4. Monitoring
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Model Monitoring")
    logger.info("=" * 80)
    
    # Setup monitoring
    monitor = ModelMonitor(
        model_name="sample_classifier",
        reference_data=X_train_scaled,
        alert_config=config.get("monitoring.alert_thresholds", {})
    )
    
    # Monitor predictions
    y_pred_test = prediction_service.predict("sample_classifier", X_test_scaled)
    monitor.monitor_prediction(X_test_scaled, y_test.values, y_pred_test)
    
    # Get monitoring report
    report = monitor.get_monitoring_report()
    logger.info(f"Monitoring report: {report}")
    
    # Save monitoring logs
    monitor.performance_monitor.save_logs()
    
    logger.info("\n" + "=" * 80)
    logger.info("MLOps Pipeline Completed Successfully!")
    logger.info("=" * 80)
    logger.info("\nPipeline Summary:")
    logger.info(f"- Data samples processed: {len(df)}")
    logger.info(f"- Model trained: {model.model_name}")
    logger.info(f"- Test accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
    logger.info(f"- Model deployed to: {config.get('deployment.default_environment', 'production')}")
    logger.info(f"- Monitoring active: Yes")


if __name__ == "__main__":
    main()
