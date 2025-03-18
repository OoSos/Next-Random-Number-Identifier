"""
Script to analyze setup.py and check if all settings are in pyproject.toml
"""

import os
import ast
import sys
import toml
from pathlib import Path

def parse_setup_py(file_path):
    """Extract setup parameters from setup.py"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    setup_params = {}
    
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'setup':
                for keyword in node.keywords:
                    if isinstance(keyword.value, ast.Constant):
                        setup_params[keyword.arg] = keyword.value.value
                    elif isinstance(keyword.value, ast.List):
                        values = []
                        for elt in keyword.value.elts:
                            if isinstance(elt, ast.Constant):
                                values.append(elt.value)
                        setup_params[keyword.arg] = values
                    elif isinstance(keyword.value, ast.Dict):
                        dict_value = {}
                        for i in range(len(keyword.value.keys)):
                            key = keyword.value.keys[i]
                            value = keyword.value.values[i]
                            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                                dict_value[key.value] = value.value
                        setup_params[keyword.arg] = dict_value
    except SyntaxError:
        print(f"Syntax error in {file_path}")
        return {}
    
    return setup_params

def read_pyproject_toml(file_path):
    """Read pyproject.toml file"""
    try:
        return toml.load(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def compare_configs(setup_params, pyproject_config):
    """Compare setup.py parameters with pyproject.toml"""
    project_config = pyproject_config.get('project', {})
    
    missing_params = []
    
    # Check name
    if 'name' in setup_params and setup_params['name'] != project_config.get('name'):
        missing_params.append(('name', setup_params['name']))
    
    # Check version
    if 'version' in setup_params and setup_params['version'] != project_config.get('version'):
        missing_params.append(('version', setup_params['version']))
    
    # Check description
    if 'description' in setup_params and setup_params['description'] != project_config.get('description')):
        missing_params.append(('description', setup_params['description']))
    
    # Check install_requires vs dependencies
    if 'install_requires' in setup_params:
        deps = project_config.get('dependencies', [])
        for req in setup_params['install_requires']:
            if not any(req in dep for dep in deps):
                missing_params.append(('dependency', req))
    
    # Check extras_require vs optional-dependencies
    if 'extras_require' in setup_params:
        for extra_name, extra_deps in setup_params['extras_require'].items():
            optional_deps = project_config.get('optional-dependencies', {}).get(extra_name, [])
            for dep in extra_deps:
                if dep not in optional_deps:
                    missing_params.append((f'optional dependency ({extra_name})', dep))
    
    return missing_params

def main():
    project_root = Path(__file__).parent.parent
    setup_py_path = project_root / 'setup.py'
    pyproject_path = project_root / 'pyproject.toml'
    
    if not setup_py_path.exists():
        print("setup.py not found.")
        return
    
    if not pyproject_path.exists():
        print("pyproject.toml not found.")
        return
    
    print("Analyzing configuration files...")
    setup_params = parse_setup_py(setup_py_path)
    pyproject_config = read_pyproject_toml(pyproject_path)
    
    missing_params = compare_configs(setup_params, pyproject_config)
    
    if missing_params:
        print("\nThe following items from setup.py are missing in pyproject.toml:")
        for param_type, param_value in missing_params:
            print(f"  - {param_type}: {param_value}")
        
        print("\nAdd these to your pyproject.toml before removing setup.py")
    else:
        print("\nAll setup.py parameters appear to be in pyproject.toml.")
        print("It should be safe to remove setup.py after manual verification.")

if __name__ == "__main__":
    main()
