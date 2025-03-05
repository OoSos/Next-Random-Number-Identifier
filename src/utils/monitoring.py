import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
import logging


class ModelMonitor:
    """
    Utility class for monitoring model performance over time and detecting drift.
    """
    
    def __init__(self, 
                drift_threshold: float = 0.1, 
                log_level: int = logging.INFO):
        """
        Initialize the ModelMonitor.
        
        Args:
            drift_threshold: Threshold for detecting significant drift
            log_level: Logging level
        """
        self.drift_threshold = drift_threshold
        self.baseline_metrics: Dict[str, float] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
    def set_baseline(self, metrics: Dict[str, float]) -> None:
        """
        Set the baseline metrics for drift detection.
        
        Args:
            metrics: Dictionary of baseline performance metrics
        """
        self.baseline_metrics = metrics.copy()
        self.logger.info(f"Baseline metrics set: {self.baseline_metrics}")
        
    def track_performance(self, 
                        metrics: Dict[str, float], 
                        timestamp: Optional[datetime] = None) -> None:
        """
        Track model performance over time.
        
        Args:
            metrics: Dictionary of performance metrics
            timestamp: Optional timestamp for the metrics (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        record = {
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        self.performance_history.append(record)
        self.logger.debug(f"Performance tracked: {metrics}")
        
        # Check for drift if baseline is set
        if self.baseline_metrics:
            self._check_drift(metrics)
            
    def _check_drift(self, current_metrics: Dict[str, float]) -> None:
        """
        Check for model drift based on current metrics vs baseline.
        
        Args:
            current_metrics: Dictionary of current performance metrics
        """
        drift_detected = False
        drift_metrics = {}
        
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                relative_change = abs(current_value - baseline_value) / (abs(baseline_value) if baseline_value != 0 else 1.0)
                
                if relative_change > self.drift_threshold:
                    drift_detected = True
                    drift_metrics[metric] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_pct': relative_change * 100
                    }
        
        if drift_detected:
            alert = {
                'timestamp': datetime.now(),
                'drift_metrics': drift_metrics,
                'severity': self._calculate_severity(drift_metrics)
            }
            self.alerts.append(alert)
            self.logger.warning(f"Drift detected: {alert}")
    
    def _calculate_severity(self, drift_metrics: Dict[str, Dict[str, float]]) -> str:
        """
        Calculate the severity of drift.
        
        Args:
            drift_metrics: Dictionary of metrics experiencing drift
            
        Returns:
            String indicating severity level ('low', 'medium', or 'high')
        """
        max_change = max(m['change_pct'] for m in drift_metrics.values())
        
        if max_change > 50:
            return 'high'
        elif max_change > 25:
            return 'medium'
        else:
            return 'low'
            
    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get a summary of drift detection results.
        
        Returns:
            Dictionary with drift detection summary
        """
        if not self.alerts:
            return {'drift_detected': False}
            
        return {
            'drift_detected': True,
            'alert_count': len(self.alerts),
            'latest_alert': self.alerts[-1],
            'highest_severity': max(alert['severity'] for alert in self.alerts),
        }
        
    def get_performance_trend(self, metric: str, window: int = -1) -> Optional[pd.Series]:
        """
        Get a time series of a specific performance metric.
        
        Args:
            metric: Name of the metric to trend
            window: Number of historical points to include (-1 for all)
            
        Returns:
            Time series of the specified metric
        """
        if not self.performance_history:
            return None
            
        # Extract the specified metric from history
        history = self.performance_history
        if window > 0:
            history = history[-window:]
            
        timestamps = [record['timestamp'] for record in history if metric in record['metrics']]
        values = [record['metrics'][metric] for record in history if metric in record['metrics']]
        
        if not timestamps:
            return None
            
        return pd.Series(values, index=timestamps)
        
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive monitoring report.
        
        Returns:
            Dictionary with monitoring report data
        """
        if not self.performance_history:
            return {'status': 'No monitoring data available'}
            
        latest_metrics = self.performance_history[-1]['metrics']
        
        # Calculate trends
        trends = {}
        for metric in latest_metrics:
            series = self.get_performance_trend(metric)
            if series is not None and len(series) > 1:
                trends[metric] = {
                    'change': float(series.iloc[-1] - series.iloc[0]),
                    'pct_change': float((series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100) if series.iloc[0] != 0 else float('inf')
                }
        
        return {
            'latest_metrics': latest_metrics,
            'trends': trends,
            'baseline_comparison': self._compare_to_baseline(latest_metrics),
            'drift_summary': self.get_drift_summary(),
            'data_points': len(self.performance_history)
        }
    
    def _compare_to_baseline(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Compare current metrics to baseline.
        
        Args:
            current_metrics: Dictionary of current metrics
            
        Returns:
            Dictionary of relative changes from baseline
        """
        if not self.baseline_metrics:
            return {}
            
        comparison = {}
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                if baseline_value != 0:
                    comparison[metric] = (current_value - baseline_value) / baseline_value * 100
                else:
                    comparison[metric] = float('inf') if current_value > 0 else 0.0
                    
        return comparison