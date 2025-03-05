import logging
import pandas as pd

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path):
        """Load data from CSV file with proper validation."""
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df.sort_values('Date')
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
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