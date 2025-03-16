import argparse
from src.utils.enhanced_data_loader import EnhancedDataLoader


def main():
    parser = argparse.ArgumentParser(description='Validate and profile a dataset')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--filename', type=str, required=True, help='Dataset filename')
    args = parser.parse_args()
    
    loader = EnhancedDataLoader(args.data_dir)
    df = loader.load_csv(args.filename)
    processed = loader.preprocess_data(df)
    validation = loader.validate_data(processed)
    profile = loader.get_data_profile(processed)
    
    print(f"Dataset: {args.filename}")
    print(f"Valid: {validation['valid']}")
    print(f"Rows: {profile['basic_info']['rows']}")
    print(f"Columns: {profile['basic_info']['columns']}")
    print(f"Missing values: {profile['basic_info']['missing_cells']}")
    
    # Print warnings and errors
    for component, results in validation.items():
        if component == 'valid':
            continue
        if 'warnings' in results:
            for warning in results['warnings']:
                print(f"Warning ({component}): {warning}")
        if 'errors' in results:
            for error in results['errors']:
                print(f"Error ({component}): {error}")
    
if __name__ == '__main__':
    main()