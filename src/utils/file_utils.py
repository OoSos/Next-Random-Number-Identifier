import os
from pathlib import Path

def debug_file_path(file_name="historical_random_numbers.csv"):
    """
    Debug function to check file paths and verify CSV file accessibility.
    
    Args:
        file_name (str): Name of the CSV file to check
        
    Returns:
        dict: Dictionary with path information
    """
    project_root = Path(__file__).parent.parent.parent.absolute()
    data_dir = project_root / "data"
    file_path = data_dir / file_name
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root directory: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"CSV file path: {file_path}")
    print(f"Data directory exists: {data_dir.exists()}")
    print(f"CSV file exists: {file_path.exists()}")
    
    result = {
        "cwd": os.getcwd(),
        "project_root": str(project_root),
        "data_dir": str(data_dir),
        "file_path": str(file_path),
        "data_dir_exists": data_dir.exists(),
        "file_exists": file_path.exists(),
        "file_content": None
    }
    
    if file_path.exists():
        # Get file size
        file_size = file_path.stat().st_size
        print(f"CSV file size: {file_size} bytes")
        result["file_size"] = file_size
        
        # Read first few lines
        try:
            with open(file_path, 'r') as f:
                head = [next(f) for _ in range(5) if f]
            print(f"CSV file first 5 lines: {head}")
            result["file_content"] = head
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            result["error"] = str(e)
    
    return result
