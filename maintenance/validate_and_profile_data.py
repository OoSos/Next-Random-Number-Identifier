import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Now this import will work
from src.utils.enhanced_data_loader import EnhancedDataLoader

loader = EnhancedDataLoader("data")
df = loader.load_csv("historical_random_numbers.csv")
processed = loader.preprocess_data(df)
validation = loader.validate_data(processed)
profile = loader.get_data_profile(processed)

print(f"Data validation: {validation['valid']}")
print(f"Data quality summary:")
print(f"  - Rows: {profile['basic_info']['rows']}")
print(f"  - Missing values: {profile['basic_info']['missing_cells']}")
print(f"  - Missing percentage: {profile['basic_info']['missing_percentage']:.2f}%")

if 'date_sequence' in validation:
    date_stats = validation['date_sequence']['stats']
    print(f"Date sequence information:")
    print(f"  - Range: {date_stats['min_date']} to {date_stats['max_date']}")
    print(f"  - Chronological order: {date_stats['is_chronological']}")
    if 'duplicate_dates' in date_stats:
        print(f"  - Duplicate dates: {date_stats['duplicate_dates']}")
    if 'large_gaps' in date_stats:
        print(f"  - Large gaps (>7 days): {date_stats['large_gaps']}")

if 'number_sequence' in validation:
    number_stats = validation['number_sequence']['stats']
    print(f"Number sequence information:")
    print(f"  - Range: {number_stats['min']} to {number_stats['max']}")
    print(f"  - Mean: {number_stats['mean']:.2f}")
    print(f"  - Median: {number_stats['median']}")
    if 'outliers' in number_stats:
        print(f"  - Outliers: {number_stats['outliers']}")