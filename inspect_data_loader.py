# inspect_data_loader.py
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.enhanced_data_loader import EnhancedDataLoader
    
    print("\nInspecting DataLoader class:")
    print(f"DataLoader methods: {[method for method in dir(DataLoader) if not method.startswith('_')]}")
    
    # Create an instance and check its methods
    data_loader = EnhancedDataLoader("data")
    print(f"\nDataLoader instance methods: {[method for method in dir(data_loader) if not method.startswith('_')]}")
    
except ImportError as e:
    print(f"Import error: {str(e)}")
    print("Looking for alternatives...")
    
    # Try to find alternative DataLoader implementations
    import importlib
    
    possible_modules = [
        'src.utils.data_loader',
        'src.data.loader',
        'src.data_loader',
        'utils.data_loader'
    ]
    
    for module_name in possible_modules:
        try:
            module = importlib.import_module(module_name)
            print(f"Found module: {module_name}")
            if hasattr(module, 'DataLoader'):
                print(f"  Contains DataLoader class")
                data_loader_class = getattr(module, 'DataLoader')
                print(f"  Methods: {[m for m in dir(data_loader_class) if not m.startswith('_')]}")
        except ImportError:
            print(f"Could not import: {module_name}")

if __name__ == "__main__":
    print("Inspecting DataLoader implementation...")