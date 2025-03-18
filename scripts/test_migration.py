"""
Script to help migrate unique tests from files being removed to consolidated files.
"""

import os
import ast
import sys
from pathlib import Path

def get_test_functions(file_path):
    """Extract all test function names and their content from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        test_functions = {}
        imports = []
        
        # Get imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                start_line = node.lineno - 1
                end_line = getattr(node, 'end_lineno', start_line) if hasattr(node, 'end_lineno') else start_line
                import_text = '\n'.join(content.split('\n')[start_line:end_line+1])
                imports.append(import_text)
        
        # Get test functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else None
                
                if end_line:
                    func_lines = content.split('\n')[start_line:end_line]
                else:
                    # Fallback for Python < 3.8
                    func_lines = []
                    in_func = False
                    indent = 0
                    for i, line in enumerate(content.split('\n')[start_line:]):
                        if i == 0:
                            # First line - get the indentation level
                            indent = len(line) - len(line.lstrip())
                            in_func = True
                            func_lines.append(line)
                        elif in_func:
                            if line.strip() == '' or line.startswith(' ' * (indent + 1)):
                                func_lines.append(line)
                            else:
                                # We've reached the end of the function
                                break
                
                test_functions[node.name] = '\n'.join(func_lines)
        
        return test_functions, imports
    except SyntaxError:
        print(f"Syntax error in {file_path}")
        return {}, []

def migrate_tests(source_files, target_file, test_dir):
    """Migrate test functions from source files to target file."""
    source_tests = {}
    all_imports = set()
    
    # Get all test functions from source files
    for source_file in source_files:
        source_path = os.path.join(test_dir, source_file)
        if not os.path.exists(source_path):
            print(f"Source file {source_path} not found.")
            continue
        
        tests, imports = get_test_functions(source_path)
        source_tests[source_file] = tests
        all_imports.update(imports)
    
    # Get existing tests in target file
    target_path = os.path.join(test_dir, target_file)
    if not os.path.exists(target_path):
        print(f"Target file {target_path} not found.")
        return
    
    target_tests, target_imports = get_test_functions(target_path)
    
    # Read target file content
    with open(target_path, 'r') as f:
        target_content = f.read()
    
    # Generate migration report
    print(f"\n=== Test Migration Report ===")
    print(f"Target file: {target_file}")
    
    for source_file, tests in source_tests.items():
        print(f"\nFrom {source_file}:")
        for test_name, test_content in tests.items():
            if test_name in target_tests:
                print(f"  - {test_name} (already exists in target)")
            else:
                print(f"  - {test_name} (will be migrated)")
    
    # Add imports that don't exist in target
    new_imports = []
    target_import_text = '\n'.join(target_imports)
    for imp in all_imports:
        if imp not in target_import_text:
            new_imports.append(imp)
    
    # Generate the migration code
    if new_imports or any(len(tests) > 0 for tests in source_tests.values()):
        print("\nMigration code to add to the target file:")
        print("```python")
        if new_imports:
            print("# Add these imports if they don't exist")
            for imp in new_imports:
                print(imp)
            print()
        
        print("# Migrated tests from source files")
        for source_file, tests in source_tests.items():
            for test_name, test_content in tests.items():
                if test_name not in target_tests:
                    print(f"\n# Migrated from {source_file}")
                    print(test_content)
        print("```")
        
        # Create a backup of the target file
        backup_path = target_path + '.bak'
        print(f"\nCreated backup of target file at {backup_path}")
        with open(backup_path, 'w') as f:
            f.write(target_content)
    else:
        print("\nNo tests need to be migrated.")

def main():
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests')
    
    migration_map = {
        'test_data_loader_consolidated.py': [
            'test_enhanced_data_loader.py',
            'test_simple_loader.py',
            'test_data_loader.py',
            'test_basic_functionality.py'
        ],
        'test_models.py': ['test_components.py'],
        'test_integration.py': ['test_pipeline.py']
    }
    
    for target_file, source_files in migration_map.items():
        migrate_tests(source_files, target_file, test_dir)

if __name__ == "__main__":
    main()
