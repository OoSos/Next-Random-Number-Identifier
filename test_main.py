"""
Test script for the main module
"""
try:
    from src.main import main
    
    result = main()
    print(f"Main function executed successfully. Result: {result}")
except ImportError as e:
    print(f"Import error: {e}")
    print("Possible solutions:")
    print("1. Make sure the 'src' directory is in your Python path")
    print("2. Ensure that the project structure is correct")
    print("3. Check if there's an __init__.py file in the src directory")
except Exception as e:
    print(f"Error while executing the main function: {e}")

if __name__ == "__main__":
    print("Running test_main.py directly...")
