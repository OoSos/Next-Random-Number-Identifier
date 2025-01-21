import os
import shutil
from pathlib import Path

def create_directory_structure():
    # Get the project root directory (where this script is located)
    project_root = Path(__file__).parent.absolute()
    
    print(f"Creating project structure in: {project_root}")
    
    # Define the directory structure
    directories = [
        Path('src/models'),
        Path('src/features'),
        Path('src/utils'),
        Path('src/visualization'),
        Path('tests/models'),
        Path('tests/features'),
        Path('tests/utils'),
        Path('tests/visualization')
    ]
    
    # Create directories
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py in each Python package directory
        if str(dir_path).startswith(('src', 'tests')):
            init_file = full_path / '__init__.py'
            init_file.touch(exist_ok=True)
            print(f"Created: {init_file}")

    # Define the files to create
    files = {
        Path('src/models/base_model.py'): '',
        Path('src/models/random_forest.py'): '',
        Path('src/models/xgboost_model.py'): '',
        Path('src/models/markov_chain.py'): '',
        Path('src/models/ensemble.py'): '',
        Path('src/features/feature_engineering.py'): '',
        Path('src/features/feature_selection.py'): '',
        Path('src/utils/data_loader.py'): '',
        Path('src/utils/evaluation.py'): '',
        Path('src/utils/monitoring.py'): '',
        Path('src/visualization/plots.py'): ''
    }
    
    # Create the files
    for file_path, content in files.items():
        full_path = project_root / file_path
        if not full_path.exists():
            full_path.write_text(content)
            print(f"Created: {full_path}")
        else:
            print(f"File already exists: {full_path}")

    # Handle existing random_number_forecast.py
    old_file = project_root / 'src' / 'random_number_forecast.py'
    if old_file.exists():
        backup_file = project_root / 'src' / 'random_number_forecast.py.bak'
        shutil.copy2(old_file, backup_file)
        print(f"\nBacked up existing random_number_forecast.py to: {backup_file}")

    print("\nDirectory structure created successfully!")
    print("\nNext steps:")
    print("1. Review the created directory structure")
    print("2. Copy the provided base_model.py content to src/models/base_model.py")
    print("3. Copy the provided data_loader.py content to src/utils/data_loader.py")
    print("4. Begin implementing other modules")

if __name__ == "__main__":
    create_directory_structure()