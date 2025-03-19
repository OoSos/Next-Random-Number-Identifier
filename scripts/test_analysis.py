"""
Script to analyze test files and identify unique test cases.
Run this before consolidating test files to ensure no tests are lost.
"""

import os
import ast
import sys
from collections import defaultdict

def get_test_functions(file_path):
    """Extract all test function names and their content from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        test_functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                # Get the source lines for this function
                lineno = node.lineno
                end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else None
                
                if end_lineno:
                    func_lines = content.split('\n')[lineno-1:end_lineno]
                else:
                    # If end_lineno is not available (Python < 3.8), approximate
                    func_lines = []
                    for line in content.split('\n')[lineno-1:]:
                        func_lines.append(line)
                        if line.strip() == '' and len(func_lines) > 1:
                            break
                
                test_functions[node.name] = '\n'.join(func_lines)
        
        return test_functions
    except SyntaxError:
        print(f"Syntax error in {file_path}")
        return {}

def analyze_test_files(test_dir):
    """Analyze all test files in the specified directory."""
    all_tests = {}
    file_tests = {}
    
    for file_name in os.listdir(test_dir):
        if not file_name.startswith('test_') or not file_name.endswith('.py'):
            continue
        
        file_path = os.path.join(test_dir, file_name)
        test_functions = get_test_functions(file_path)
        file_tests[file_name] = test_functions
        
        for test_name, test_content in test_functions.items():
            if test_name in all_tests:
                all_tests[test_name].append(file_name)
            else:
                all_tests[test_name] = [file_name]
    
    return all_tests, file_tests

def identify_unique_tests(to_remove, to_keep, all_tests, file_tests):
    """Identify tests unique to files being removed."""
    unique_tests = {}
    
    for file in to_remove:
        if file not in file_tests:
            continue
        
        unique_tests[file] = []
        for test_name in file_tests[file]:
            is_unique = True
            for keep_file in to_keep:
                if keep_file in file_tests and test_name in file_tests[keep_file]:
                    is_unique = False
                    break
            
            if is_unique:
                unique_tests[file].append(test_name)
    
    return unique_tests

def main():
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests')
    
    to_remove = [
        'test_enhanced_data_loader.py',
        'test_components.py',
        'test_simple_loader.py',
        'test_basic_functionality.py',
        'test_data_loader.py',
        'test_pipeline.py'
    ]
    
    to_keep = [
        'test_data_loader_consolidated.py',
        'test_models.py',
        'test_integration.py'
    ]
    
    print(f"Analyzing test files in {test_dir}\n")
    all_tests, file_tests = analyze_test_files(test_dir)
    
    # Print duplicate tests
    duplicate_tests = {name: files for name, files in all_tests.items() if len(files) > 1}
    print(f"Found {len(duplicate_tests)} tests with duplicates:")
    for test_name, files in duplicate_tests.items():
        print(f"  {test_name}: {', '.join(files)}")
    
    # Print unique tests that will be lost
    unique_tests = identify_unique_tests(to_remove, to_keep, all_tests, file_tests)
    print("\nUnique tests in files to be removed:")
    for file, tests in unique_tests.items():
        if tests:
            print(f"\n{file} ({len(tests)} unique tests):")
            for test_name in tests:
                print(f"  - {test_name}")
                
                # Print the first few lines of the test function
                content = file_tests[file][test_name]
                preview = '\n    '.join(content.split('\n')[:3]) + '...'
                print(f"    {preview}")
    
    # Provide guidance on migration
    print("\n=== Migration Guidance ===")
    total_unique = sum(len(tests) for tests in unique_tests.values())
    if total_unique > 0:
        print(f"Found {total_unique} unique tests that need to be migrated before removing files.")
        print("Recommended actions:")
        print("1. Copy these unique tests to the corresponding consolidated files:")
        migration_map = {
            'test_enhanced_data_loader.py': 'test_data_loader_consolidated.py',
            'test_simple_loader.py': 'test_data_loader_consolidated.py',
            'test_data_loader.py': 'test_data_loader_consolidated.py',
            'test_components.py': 'test_models.py',
            'test_pipeline.py': 'test_integration.py',
            'test_basic_functionality.py': 'test_data_loader_consolidated.py'
        }
        for file, target in migration_map.items():
            if file in unique_tests and unique_tests[file]:
                print(f"   - Migrate tests from {file} to {target}")
    else:
        print("No unique tests found. Safe to remove the redundant files.")

if __name__ == "__main__":
    main()
