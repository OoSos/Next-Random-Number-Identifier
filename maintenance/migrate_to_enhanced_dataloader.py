#!/usr/bin/env python3
"""
Migration script to help transition from old DataLoader to EnhancedDataLoader.
This script:
1. Analyzes imports across the codebase
2. Suggests files that need updating
3. Can automatically update imports if requested
"""

import os
import re
from pathlib import Path
import argparse

def find_dataloader_usages(directory, patterns=None):
    """Find all files using DataLoader."""
    if patterns is None:
        patterns = [
            r'from\s+.*\.data_loader\s+import\s+DataLoader',
            r'from\s+.*utils\s+import\s+DataLoader',
            r'data_loader\s*=\s*DataLoader\(',
            r'loader\s*=\s*DataLoader\('
        ]
    
    matches = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern in patterns:
                        if re.search(pattern, content):
                            matches.append((file_path, pattern))
                            break
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return matches

def suggest_updates(matches):
    """Suggest updates for each file."""
    suggestions = {}
    
    for file_path, pattern in matches:
        if 'from' in pattern:
            # Import pattern
            suggestions[file_path] = "Replace 'from ... import DataLoader' with 'from src.utils.enhanced_data_loader import EnhancedDataLoader'"
        else:
            # Usage pattern
            suggestions[file_path] = "Replace 'DataLoader(' with 'EnhancedDataLoader('"
    
    return suggestions

def update_file(file_path, dry_run=True):
    """Update a file to use EnhancedDataLoader."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace imports
        updated_content = re.sub(
            r'from\s+(.*?)\.data_loader\s+import\s+DataLoader',
            r'from src.utils.enhanced_data_loader import EnhancedDataLoader',
            content
        )
        
        # Replace from utils import
        updated_content = re.sub(
            r'from\s+(.*?)\.utils\s+import\s+DataLoader',
            r'from \1.utils import EnhancedDataLoader',
            updated_content
        )
        
        # Replace instantiations
        updated_content = re.sub(
            r'(\w+)\s*=\s*DataLoader\(',
            r'\1 = EnhancedDataLoader(',
            updated_content
        )
        
        if dry_run:
            print(f"Would update {file_path}")
            return content != updated_content
        else:
            if content != updated_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                print(f"Updated {file_path}")
                return True
            else:
                print(f"No changes needed for {file_path}")
                return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Migrate from DataLoader to EnhancedDataLoader')
    parser.add_argument('--directory', type=str, default='src', help='Directory to search')
    parser.add_argument('--update', action='store_true', help='Actually update files')
    args = parser.parse_args()
    
    print(f"Searching for DataLoader usages in {args.directory}...")
    matches = find_dataloader_usages(args.directory)
    
    if not matches:
        print("No DataLoader usages found.")
        return
    
    print(f"Found {len(matches)} files using DataLoader:")
    suggestions = suggest_updates(matches)
    
    for file_path, suggestion in suggestions.items():
        print(f"\n{file_path}")
        print(f"  Suggestion: {suggestion}")
    
    if args.update:
        print("\nUpdating files...")
        updated_count = sum(update_file(file_path, dry_run=False) for file_path, _ in matches)
        print(f"\nUpdated {updated_count} files.")
    else:
        print("\nRun with --update to apply changes.")

if __name__ == "__main__":
    main()