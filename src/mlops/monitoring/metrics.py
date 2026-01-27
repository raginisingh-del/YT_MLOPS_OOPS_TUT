"""Model monitoring and performance tracking module."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitors model performance over time."""
    
    def __init__(self, model_name: str, log_dir: Optional[Path] = None):
        """
        Initialize PerformanceMonitor.
        
        Args:
            model_name: Name of the model being monitored
            log_dir: Directory to store monitoring logs
        """
        self.model_name = model_name
        self.log_dir = Path(log_dir) if log_dir else Path("logs/monitoring")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[Dict[str, Any]] = []
        
    def log_prediction(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      timestamp: Optional[datetime] = None):
        """
        Log a prediction for monitoring.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            timestamp: Timestamp of prediction
        """
        timestamp = timestamp or datetime.now()
        
        metrics = {
            "timestamp": timestamp,
            "accuracy": accuracy_score(y_true, y_pred),
            "samples": len(y_true),
        }
        
        self.metrics_history.append(metrics)
        logger.info(f"Logged prediction metrics: {metrics}")
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        return metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of monitored metrics.
        
        Returns:
            Summary statistics
        """
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        summary = {
            "mean_accuracy": df["accuracy"].mean(),
            "std_accuracy": df["accuracy"].std(),
            "min_accuracy": df["accuracy"].min(),
            "max_accuracy": df["accuracy"].max(),
            "total_predictions": df["samples"].sum(),
        }
        return summary
    
    def save_logs(self):
        """Save monitoring logs to disk."""
        if not self.metrics_history:
            logger.warning("No metrics to save")
            return
            
        log_file = self.log_dir / f"{self.model_name}_metrics.csv"
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(log_file, index=False)
        logger.info(f"Monitoring logs saved to {log_file}")


class DataDriftDetector:
    """Detects data drift in production data."""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.1):
        """
        Initialize DataDriftDetector.
        
        Args:
            reference_data: Reference dataset (e.g., training data)
            threshold: Drift threshold
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats = self._calculate_statistics(reference_data)
        
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics for data.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary of statistics
        
        Note:
            Uses simple mean/std comparison for drift detection.
            For production use, consider more sophisticated methods like
            Kolmogorov-Smirnov test or Population Stability Index (PSI).
        """
        stats = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                "mean": data[col].mean(),
                "std": data[col].std(),
                "min": data[col].min(),
                "max": data[col].max(),
            }
        return stats
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect drift in new data.
        
        Args:
            new_data: New data to check for drift
            
        Returns:
            Dictionary indicating drift per feature
        """
        new_stats = self._calculate_statistics(new_data)
        drift_detected = {}
        
        for col in self.reference_stats:
            if col not in new_stats:
                continue
                
            # Simple drift detection based on mean shift
            ref_mean = self.reference_stats[col]["mean"]
            new_mean = new_stats[col]["mean"]
            ref_std = self.reference_stats[col]["std"]
            
            if ref_std > 0:
                drift = abs(new_mean - ref_mean) / ref_std
                drift_detected[col] = drift > self.threshold
            else:
                drift_detected[col] = False
                
        logger.info(f"Drift detection completed. Drifted features: {sum(drift_detected.values())}")
        return drift_detected


class AlertManager:
    """Manages alerts for monitoring events."""
    
    def __init__(self, alert_config: Optional[Dict[str, Any]] = None):
        """
        Initialize AlertManager.
        
        Args:
            alert_config: Configuration for alerts
        """
        self.alert_config = alert_config or {}
        self.alerts: List[Dict[str, Any]] = []
        
    def check_thresholds(self, metrics: Dict[str, float]):
        """
        Check if metrics exceed thresholds.
        
        Args:
            metrics: Current metrics
        """
        for metric_name, value in metrics.items():
            threshold_key = f"{metric_name}_threshold"
            if threshold_key in self.alert_config:
                threshold = self.alert_config[threshold_key]
                if value < threshold:
                    self.trigger_alert(
                        f"{metric_name} below threshold",
                        f"{metric_name}={value:.4f} < {threshold}"
                    )
                    
    def trigger_alert(self, alert_type: str, message: str):
        """
        Trigger an alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
        """
        alert = {
            "timestamp": datetime.now(),
            "type": alert_type,
            "message": message,
        }
        self.alerts.append(alert)
        logger.warning(f"ALERT: {alert_type} - {message}")
        
    def get_alerts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        if limit:
            return self.alerts[-limit:]
        return self.alerts


class ModelMonitor:
    """Comprehensive model monitoring system."""
    
    def __init__(self, model_name: str, reference_data: pd.DataFrame,
                 alert_config: Optional[Dict[str, Any]] = None):
        """
        Initialize ModelMonitor.
        
        Args:
            model_name: Name of the model
            reference_data: Reference dataset
            alert_config: Alert configuration
        """
        self.model_name = model_name
        self.performance_monitor = PerformanceMonitor(model_name)
        self.drift_detector = DataDriftDetector(reference_data)
        self.alert_manager = AlertManager(alert_config)
        
    def monitor_prediction(self, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Monitor a prediction.
        
        Args:
            X: Input features
            y_true: True labels
            y_pred: Predicted labels
        """
        # Log performance
        self.performance_monitor.log_prediction(y_true, y_pred)
        
        # Check for data drift
        drift_detected = self.drift_detector.detect_drift(X)
        if any(drift_detected.values()):
            drifted_features = [f for f, d in drift_detected.items() if d]
            self.alert_manager.trigger_alert(
                "Data Drift",
                f"Drift detected in features: {', '.join(drifted_features)}"
            )
            
        # Check performance thresholds
        metrics = self.performance_monitor.calculate_metrics(y_true, y_pred)
        self.alert_manager.check_thresholds(metrics)
        
    def get_monitoring_report(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring report.
        
        Returns:
            Monitoring report
        """
        report = {
            "model_name": self.model_name,
            "performance_summary": self.performance_monitor.get_metrics_summary(),
            "recent_alerts": self.alert_manager.get_alerts(limit=10),
            "timestamp": datetime.now(),
        }
        return report
