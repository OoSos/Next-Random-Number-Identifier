from utils.monitoring import ModelMonitor

def setup_monitoring(metrics=None):
    """
    Setup and run the model monitoring pipeline.
    
    Parameters:
    -----------
    metrics : dict
        Initial performance metrics to set as baseline
        
    Returns:
    --------
    monitor : ModelMonitor
        Configured model monitor object
    """
    # Initialize monitor
    monitor = ModelMonitor(drift_threshold=0.15)
    
    # Set baseline metrics if provided
    if metrics:
        monitor.set_baseline(metrics)
        
    return monitor

def run_monitoring_cycle(monitor, new_metrics):
    """
    Run a monitoring cycle to check for model drift.
    
    Parameters:
    -----------
    monitor : ModelMonitor
        The initialized monitor object
    new_metrics : dict
        New performance metrics to compare against baseline
        
    Returns:
    --------
    dict
        Monitoring results including drift status
    """
    # Track performance over time
    monitor.track_performance(new_metrics)
    
    # Generate monitoring report
    report = monitor.generate_report()
    print("Model monitoring report:", report)
    
    # Check for drift
    drift_summary = monitor.get_drift_summary()
    if drift_summary['drift_detected']:
        print("Warning: Model drift detected")
        print("Drift details:", drift_summary['latest_alert'])
        
    return {
        "report": report,
        "drift_summary": drift_summary
    }