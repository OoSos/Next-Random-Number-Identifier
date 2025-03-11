import argparse
import pandas as pd  # Standard import convention for pandas
from main import main
from utils.data_loader import DataLoader
from utils.monitoring_pipeline import setup_monitoring, run_monitoring_cycle

def parse_args():
    parser = argparse.ArgumentParser(description='Next Random Number Identifier')
    parser.add_argument('--data', type=str, default='data/historical_random_numbers.csv',
                       help='Path to historical data CSV')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate', 'monitor'],
                       default='predict', help='Operation mode')
    parser.add_argument('--model', type=str, choices=['rf', 'xgb', 'markov', 'ensemble', 'hybrid'],
                       default='ensemble', help='Model type to use')
    parser.add_argument('--monitor', action='store_true', help='Enable model monitoring')
    return parser.parse_args()

def clean_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Before cleaning: {df.shape}, NaN count: {df.isna().sum().sum()}")
    
    # Fill NaN values first
    df = df.ffill().bfill()  # Use the updated syntax instead of deprecated method
    
    # Then drop any remaining NaN values
    df.dropna(inplace=True)
    
    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    if cleaned_file_path == file_path:  # Ensure we don't overwrite the original file
        cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    
    df.to_csv(cleaned_file_path, index=False)
    print(f"After cleaning: {df.shape}, NaN count: {df.isna().sum().sum()}")
    return cleaned_file_path

def run_cli():
    """Run the CLI application with parsed arguments."""
    args = parse_args()
    
    cleaned_data_path = clean_data(args.data)
    
    # Run the main functionality
    if args.mode == 'train':
        print(f"Training model: {args.model} using data: {cleaned_data_path}")
        results = main()
        
        # Enable monitoring if requested
        if args.monitor:
            model = results['models'][args.model]
            monitor = setup_monitoring(results[f'{args.model}_metrics'] 
                                       if f'{args.model}_metrics' in results 
                                       else results['ensemble_metrics'])
            print(f"Model monitoring enabled for {args.model}")
    
    elif args.mode == 'predict':
        print(f"Running prediction with model: {args.model}")
        results = main()
        model = results['models'][args.model]
        data_loader = DataLoader("data")
        
        # Load new data for prediction
        df = data_loader.load_csv(cleaned_data_path)
        df = data_loader.preprocess_data(df)
        
        # Make predictions with selected model
        predictions = model.predict(df)
        print(f"Predictions: {predictions}")
        
    elif args.mode == 'evaluate':
        print(f"Evaluating model: {args.model}")
        results = main()
        metrics_key = f"{args.model}_metrics" if f"{args.model}_metrics" in results else "ensemble_metrics"
        print(f"Model performance: {results[metrics_key]}")
    
    elif args.mode == 'monitor':
        print("Running model monitoring")
        results = main()
        
        # Setup monitoring for the selected model
        monitor = setup_monitoring(results[f'{args.model}_metrics'] 
                                  if f'{args.model}_metrics' in results 
                                  else results['ensemble_metrics'])
        
        # This would typically be run with new metrics as they become available
        print("Simulating monitoring cycle with current metrics")
        run_monitoring_cycle(monitor, results[f'{args.model}_metrics'] 
                            if f'{args.model}_metrics' in results 
                            else results['ensemble_metrics'])

if __name__ == "__main__":
    run_cli()