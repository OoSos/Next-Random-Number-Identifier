#!/usr/bin/env python3
"""
Import Standardization Implementation Script

This script applies the final import standardization changes across the codebase.
It fixes the remaining import order issues identified in the validation report.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple


class ImportStandardizer:
    """Automatically standardizes imports in Python files."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.changes_made = 0
        
    def standardize_file(self, file_path: Path) -> bool:
        """Standardize imports in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip files with complex import patterns for now
            if 'sys.path' in content and 'test' in str(file_path):
                return False
                
            # Parse AST to find imports
            tree = ast.parse(content)
            imports_info = self._extract_imports(tree, content)
            
            if not imports_info:
                return False
            
            # Reorganize imports
            new_content = self._reorganize_imports(content, imports_info)
            
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                self.changes_made += 1
                return True
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
        return False
    
    def _extract_imports(self, tree: ast.AST, content: str) -> List[Dict]:
        """Extract import information from AST."""
        imports = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start_line = node.lineno - 1
                end_line = getattr(node, 'end_lineno', start_line) or start_line
                
                import_text = '\n'.join(lines[start_line:end_line + 1])
                
                # Determine import type
                if isinstance(node, ast.Import):
                    module_name = node.names[0].name.split('.')[0]
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                    else:
                        continue
                else:
                    continue
                
                # Categorize import
                category = self._categorize_import(module_name, node)
                
                imports.append({
                    'start_line': start_line,
                    'end_line': end_line,
                    'text': import_text,
                    'category': category,
                    'module': module_name
                })
        
        return sorted(imports, key=lambda x: x['start_line'])
    
    def _categorize_import(self, module_name: str, node: ast.stmt) -> str:
        """Categorize an import as stdlib, third-party, or local."""
        stdlib_modules = {
            'os', 'sys', 'logging', 'pathlib', 'typing', 'collections', 
            'datetime', 'tempfile', 'unittest', 'argparse', 'json', 'abc',
            'functools', 'itertools', 'math', 'random', 'time', 'warnings'
        }
        
        third_party_modules = {
            'pandas', 'numpy', 'sklearn', 'matplotlib', 'pytest', 
            'xgboost', 'pydantic', 'scipy', 'typing_extensions'
        }
        
        # Check for relative imports
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            return 'local_relative'
        
        if module_name in stdlib_modules:
            return 'stdlib'
        elif module_name in third_party_modules:
            return 'third_party'
        elif module_name == 'src' or module_name.startswith('src.'):
            return 'local'
        else:
            # Default to third-party for unknown modules
            return 'third_party'
    
    def _reorganize_imports(self, content: str, imports_info: List[Dict]) -> str:
        """Reorganize imports according to PEP 8."""
        if not imports_info:
            return content
        
        lines = content.split('\n')
        
        # Find the range of import lines
        first_import = min(imp['start_line'] for imp in imports_info)
        last_import = max(imp['end_line'] for imp in imports_info)
        
        # Group imports by category
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        relative_imports = []
        
        for imp in imports_info:
            if imp['category'] == 'stdlib':
                stdlib_imports.append(imp['text'])
            elif imp['category'] == 'third_party':
                third_party_imports.append(imp['text'])
            elif imp['category'] == 'local':
                local_imports.append(imp['text'])
            elif imp['category'] == 'local_relative':
                relative_imports.append(imp['text'])
        
        # Create new import section
        new_imports = []
        
        if stdlib_imports:
            new_imports.append("# Standard library imports")
            new_imports.extend(sorted(stdlib_imports))
            new_imports.append("")
        
        if third_party_imports:
            new_imports.append("# Third-party imports") 
            new_imports.extend(sorted(third_party_imports))
            new_imports.append("")
        
        if local_imports:
            new_imports.append("# Local application imports")
            new_imports.extend(sorted(local_imports))
            new_imports.append("")
        
        # Keep relative imports for __init__.py files
        if relative_imports:
            new_imports.extend(relative_imports)
            new_imports.append("")
        
        # Remove trailing empty line
        if new_imports and new_imports[-1] == "":
            new_imports.pop()
        
        # Reconstruct the file
        new_lines = lines[:first_import] + new_imports + lines[last_import + 1:]
        
        return '\n'.join(new_lines)
    
    def standardize_project(self) -> None:
        """Standardize imports across the entire project."""
        print("🔧 Starting import standardization...")
        
        # Find Python files in src directory
        src_files = list((self.project_root / "src").glob("**/*.py"))
        
        # Filter out __pycache__ and other build artifacts
        src_files = [f for f in src_files if '__pycache__' not in str(f)]
        
        for file_path in src_files:
            try:
                if self.standardize_file(file_path):
                    print(f"✅ Standardized: {file_path.relative_to(self.project_root)}")
                else:
                    print(f"⏭️  Skipped: {file_path.relative_to(self.project_root)}")
            except Exception as e:
                print(f"❌ Error with {file_path.relative_to(self.project_root)}: {e}")
        
        print(f"\n📝 Import standardization complete!")
        print(f"Files modified: {self.changes_made}")


def main():
    """Main function to run import standardization."""
    project_root = Path(__file__).parent
    
    print("🚀 Import Standardization Tool")
    print("=" * 40)
    
    standardizer = ImportStandardizer(str(project_root))
    standardizer.standardize_project()
    
    print("\n🔍 Running validation to check results...")
    
    # Run validation
    try:
        import subprocess
        result = subprocess.run([
            'python', 'validate_imports.py'
        ], cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All imports are now standardized!")
        else:
            print("⚠️  Some issues may remain. Check the validation report.")
            
    except Exception as e:
        print(f"❌ Could not run validation: {e}")


if __name__ == "__main__":
    main()
