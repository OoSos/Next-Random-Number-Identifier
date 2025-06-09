#!/usr/bin/env python3
"""
Import Pattern Validation Script

This script validates that the import standardization has been applied correctly
across the codebase. It checks for:

1. Proper import order (stdlib, third-party, local)
2. Consistent absolute import usage
3. Removal of sys.path manipulations in test files
4. Proper import grouping and spacing
"""

import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict


class ImportAnalyzer:
    """Analyzes Python files for import patterns."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues = defaultdict(list)
        self.stats = {
            'files_analyzed': 0,
            'issues_found': 0,
            'clean_files': 0
        }
    
    def analyze_file(self, file_path: Path) -> Dict[str, List[str]]:
        """Analyze a single Python file for import issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Extract imports with line numbers
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append((node.lineno, node))
            
            # Sort by line number
            imports.sort()
            
            # Check import order and patterns
            issues.extend(self._check_import_order(imports, content))
            issues.extend(self._check_relative_imports(imports, file_path))
            issues.extend(self._check_sys_path_manipulation(content))
            issues.extend(self._check_import_grouping(content))
            
        except Exception as e:
            issues.append(f"Failed to parse file: {str(e)}")
        
        return issues
    
    def _check_import_order(self, imports: List[Tuple[int, ast.stmt]], content: str) -> List[str]:
        """Check if imports follow the standard order: stdlib, third-party, local."""
        issues = []
        
        stdlib_modules = {
            'os', 'sys', 'logging', 'pathlib', 'typing', 'collections', 
            'datetime', 'tempfile', 'unittest', 'argparse', 'json'
        }
        
        third_party_modules = {
            'pandas', 'numpy', 'sklearn', 'matplotlib', 'pytest', 
            'xgboost', 'pydantic'
        }
        
        current_group = 'stdlib'
        
        for line_no, node in imports:
            if isinstance(node, ast.Import):
                module_name = node.names[0].name.split('.')[0]
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                else:
                    continue
            else:
                continue
            
            # Determine group
            if module_name in stdlib_modules:
                group = 'stdlib'
            elif module_name in third_party_modules:
                group = 'third_party'
            elif module_name == 'src' or module_name.startswith('src.'):
                group = 'local'
            else:
                # Could be either third-party or local, skip for now
                continue
            
            # Check order
            if group == 'stdlib' and current_group in ['third_party', 'local']:
                issues.append(f"Line {line_no}: Standard library import after third-party/local imports")
            elif group == 'third_party' and current_group == 'local':
                issues.append(f"Line {line_no}: Third-party import after local imports")
            
            # Update current group
            if group == 'third_party' and current_group == 'stdlib':
                current_group = 'third_party'
            elif group == 'local' and current_group in ['stdlib', 'third_party']:
                current_group = 'local'
        
        return issues
    
    def _check_relative_imports(self, imports: List[Tuple[int, ast.stmt]], file_path: Path) -> List[str]:
        """Check for relative imports (should use absolute imports instead)."""
        issues = []
        
        # Allow relative imports only in __init__.py files
        is_init_file = file_path.name == '__init__.py'
        
        for line_no, node in imports:
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                if not is_init_file:
                    issues.append(f"Line {line_no}: Relative import found, use absolute import instead")
        
        return issues
    
    def _check_sys_path_manipulation(self, content: str) -> List[str]:
        """Check for sys.path.insert or sys.path.append usage."""
        issues = []
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'sys.path.insert' in line or 'sys.path.append' in line:
                issues.append(f"Line {i}: sys.path manipulation found, use absolute imports instead")
        
        return issues
    
    def _check_import_grouping(self, content: str) -> List[str]:
        """Check for proper import grouping with blank lines."""
        issues = []
        
        lines = content.split('\n')
        import_lines = []
        
        # Find all import lines
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') or
                stripped.startswith('# Standard library') or
                stripped.startswith('# Third-party') or
                stripped.startswith('# Local application')):
                import_lines.append((i, line))
        
        # Check for comment headers
        has_stdlib_comment = any('# Standard library' in line for _, line in import_lines)
        has_third_party_comment = any('# Third-party' in line for _, line in import_lines)
        has_local_comment = any('# Local application' in line for _, line in import_lines)
        
        if len(import_lines) > 3:  # Only check grouping for files with multiple imports
            if not (has_stdlib_comment or has_third_party_comment or has_local_comment):
                issues.append("Consider adding import group comments for better organization")
        
        return issues
    
    def analyze_project(self) -> Dict[str, List[str]]:
        """Analyze all Python files in the project."""
        python_files = list(self.project_root.glob('**/*.py'))
        
        # Filter out venv and __pycache__ directories
        python_files = [
            f for f in python_files 
            if not any(part.startswith(('.venv', 'venv', '__pycache__', '.git')) 
                      for part in f.parts)
        ]
        
        all_issues = {}
        
        for file_path in python_files:
            self.stats['files_analyzed'] += 1
            issues = self.analyze_file(file_path)
            
            if issues:
                relative_path = file_path.relative_to(self.project_root)
                all_issues[str(relative_path)] = issues
                self.stats['issues_found'] += len(issues)
            else:
                self.stats['clean_files'] += 1
        
        return all_issues
    
    def generate_report(self, issues: Dict[str, List[str]]) -> str:
        """Generate a formatted report of import issues."""
        report = []
        report.append("# Import Standardization Validation Report")
        report.append(f"Generated on: {Path().cwd()}")
        report.append("")
        
        # Statistics
        report.append("## Summary Statistics")
        report.append(f"- Files analyzed: {self.stats['files_analyzed']}")
        report.append(f"- Files with issues: {len(issues)}")
        report.append(f"- Clean files: {self.stats['clean_files']}")
        report.append(f"- Total issues found: {self.stats['issues_found']}")
        report.append("")
        
        if not issues:
            report.append("🎉 **All files pass import standardization!**")
            return '\n'.join(report)
        
        # Detailed issues
        report.append("## Issues Found")
        report.append("")
        
        for file_path, file_issues in sorted(issues.items()):
            report.append(f"### `{file_path}`")
            for issue in file_issues:
                report.append(f"- {issue}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        report.append("1. **Import Order**: Follow PEP 8 - stdlib, third-party, local")
        report.append("2. **Absolute Imports**: Use `from src.module import Class` instead of relative imports")
        report.append("3. **Remove sys.path**: Avoid `sys.path.insert()` in favor of proper package structure")
        report.append("4. **Group Comments**: Add section comments for better organization")
        report.append("")
        
        return '\n'.join(report)


def main():
    """Main function to run the import analysis."""
    project_root = Path(__file__).parent
    
    print("🔍 Analyzing import patterns...")
    analyzer = ImportAnalyzer(str(project_root))
    issues = analyzer.analyze_project()
    
    print(f"📊 Analysis complete: {analyzer.stats['files_analyzed']} files analyzed")
    
    # Generate report
    report = analyzer.generate_report(issues)
    
    # Save report
    report_path = project_root / "import_validation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📝 Report saved to: {report_path}")
    
    # Print summary to console
    if issues:
        print(f"⚠️  Found issues in {len(issues)} files")
        print("See the report for details.")
    else:
        print("✅ All files pass import standardization!")
    
    return len(issues)


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
