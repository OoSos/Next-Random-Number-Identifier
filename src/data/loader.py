import logging
import pandas as pd

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path):
        """Load data from CSV file with proper validation."""
        try:
            # Updated to use on_bad_lines instead of error_bad_lines
            df = pd.read_csv(file_path, on_bad_lines='warn')
            df['Date'] = pd.to_datetime(df['Date'])
            return df.sort_values('Date')
        except Exception as e:
            self.logger.error(f"Error loading data with pandas: {str(e)}")
            self.logger.info("Attempting manual CSV loading as fallback...")
            return self.manual_csv_load(file_path)
            
    def manual_csv_load(self, file_path):
        """
        Manually load a CSV file line by line for debugging.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        self.logger.info(f"Manually loading CSV file: {file_path}")
        
        rows = []
        
        try:
            with open(file_path, 'r') as f:
                # Read header
                header = f.readline().strip().split(',')
                self.logger.info(f"Header: {header}")
                
                # Read data
                for i, line in enumerate(f):
                    try:
                        values = line.strip().split(',')
                        row = dict(zip(header, values))
                        rows.append(row)
                    except Exception as e:
                        self.logger.error(f"Error parsing line {i}: {line} - {str(e)}")
                
                self.logger.info(f"Loaded {len(rows)} rows")
        except Exception as e:
            self.logger.error(f"Error opening CSV file: {str(e)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Convert types
        if 'Number' in df.columns:
            df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            return df.sort_values('Date')
        
        return df
            
class DataValidator:
    def validate_data(self, df):
        """Comprehensive data validation."""
        validation_report = {
            'is_valid': True,
            'issues': []
        }
        
        # Check for missing values
        if df.isnull().any().any():
            validation_report['is_valid'] = False
            validation_report['issues'].append("Missing values detected")
            
        # Check for valid number range (1-10)
        if not df['Number'].between(1, 10).all():
            validation_report['is_valid'] = False
            validation_report['issues'].append("Numbers outside valid range (1-10)")
            
        # Check date continuity (not too many gaps)
        date_gaps = df['Date'].sort_values().diff().dt.days
        if date_gaps.max() > 7:  # Configurable threshold
            validation_report['issues'].append(f"Unusual gap in dates: {date_gaps.max()} days")
            
        return validation_report