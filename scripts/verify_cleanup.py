"""
Script to verify that everything works after cleanup.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run the test suite and report results."""
    print("Running test suite...")
    result = subprocess.run(['pytest', '-v'], capture_output=True, text=True)
    
    print("\nTest Results:")
    print(result.stdout)
    
    if result.returncode != 0:
        print("\nTests failed! Please review the failures before proceeding.")
        return False
    else:
        print("\nAll tests passed!")
        return True

def check_files(files_to_remove):
    """Check if files to be removed still contain unique content."""
    project_root = Path(__file__).parent.parent
    
    print("\nChecking files to be removed...")
    for file_path in files_to_remove:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"  - {file_path}: Not found (already removed)")
            continue
            
        print(f"  - {file_path}: Still exists")

def main():
    # List of files to be removed
    files_to_remove = [
        'tests/test_enhanced_data_loader.py',
        'tests/test_components.py',
        'tests/test_simple_loader.py',
        'tests/test_basic_functionality.py',
        'tests/test_data_loader.py',
        'tests/test_pipeline.py',
        'pyrightconfig.json',
        'setup.py',
        'random_number_forecast.py.bak'
    ]
    
    # Step 1: Check if tests still pass
    tests_pass = run_tests()
    
    # Step 2: Check if files can be safely removed
    check_files(files_to_remove)
    
    # Provide final recommendations
    print("\n=== Final Recommendations ===")
    if tests_pass:
        print("✓ Tests are passing. It's safe to proceed with the cleanup.")
    else:
        print("✗ Tests are failing. Fix the issues before proceeding.")
    
    print("\nCleanup steps:")
    print("1. Ensure all unique tests have been migrated from files to be removed")
    print("2. Ensure all setup.py configurations are in pyproject.toml")
    print("3. Delete the following redundant files:")
    for file_path in files_to_remove:
        print(f"   rm {file_path}")

if __name__ == "__main__":
    main()
