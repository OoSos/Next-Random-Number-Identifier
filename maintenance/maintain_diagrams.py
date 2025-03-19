#!/usr/bin/env python
"""
Diagram maintenance utility for Next Random Number Identifier project.

This script helps maintain consistency between Mermaid source files and their
PNG renderings. It can be used to:
1. Check which diagrams are out of sync with their source
2. Generate PNG files from Mermaid sources using the Mermaid CLI

Requirements:
- Node.js and npm must be installed
- @mermaid-js/mermaid-cli package (Install with: npm install -g @mermaid-js/mermaid-cli)
"""

import os
import glob
import hashlib
import subprocess
import argparse
from datetime import datetime
from pathlib import Path


def calculate_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def find_diagram_pairs(diagrams_dir):
    """Find all Mermaid source files and their corresponding PNG files."""
    diagram_pairs = []
    mermaid_files = glob.glob(os.path.join(diagrams_dir, "*.mermaid"))
    
    for mermaid_file in mermaid_files:
        base_name = os.path.splitext(mermaid_file)[0]
        png_file = f"{base_name}.png"
        
        if os.path.exists(png_file):
            diagram_pairs.append((mermaid_file, png_file))
        else:
            diagram_pairs.append((mermaid_file, None))
    
    return diagram_pairs


def check_diagram_sync(diagrams_dir):
    """Check which diagrams are out of sync with their source files."""
    diagram_pairs = find_diagram_pairs(diagrams_dir)
    out_of_sync = []
    
    print(f"\nChecking diagram synchronization in {diagrams_dir}...")
    
    for mermaid_file, png_file in diagram_pairs:
        mermaid_modified = datetime.fromtimestamp(os.path.getmtime(mermaid_file))
        
        if png_file is None:
            print(f"⚠️  Missing PNG for {os.path.basename(mermaid_file)}")
            out_of_sync.append((mermaid_file, None))
            continue
            
        png_modified = datetime.fromtimestamp(os.path.getmtime(png_file))
        
        if mermaid_modified > png_modified:
            print(f"⚠️  Out of sync: {os.path.basename(mermaid_file)} (modified after PNG)")
            out_of_sync.append((mermaid_file, png_file))
        else:
            print(f"✓ In sync: {os.path.basename(mermaid_file)}")
    
    return out_of_sync


def generate_png_from_mermaid(mermaid_file, png_file=None):
    """Generate a PNG file from a Mermaid source file using mmdc CLI."""
    if png_file is None:
        base_name = os.path.splitext(mermaid_file)[0]
        png_file = f"{base_name}.png"
    
    print(f"Generating {os.path.basename(png_file)} from {os.path.basename(mermaid_file)}...")
    
    try:
        # Use Mermaid CLI to generate the PNG
        result = subprocess.run(
            [
                "mmdc",
                "-i", mermaid_file,
                "-o", png_file,
                "-b", "transparent"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Successfully generated {os.path.basename(png_file)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating diagram: {e}")
        print(f"Error details: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Error: Mermaid CLI (mmdc) not found.")
        print("Please install it with: npm install -g @mermaid-js/mermaid-cli")
        return False


def main():
    """Main function to check and update diagrams."""
    parser = argparse.ArgumentParser(description="Manage Mermaid diagrams in the project.")
    parser.add_argument("--check", action="store_true", help="Check if diagrams are out of sync")
    parser.add_argument("--update", action="store_true", help="Update out-of-sync diagrams")
    parser.add_argument("--all", action="store_true", help="Update all diagrams regardless of sync status")
    parser.add_argument("--dir", default="./docs/diagrams", help="Directory containing diagram files (default: ./docs/diagrams)")
    
    args = parser.parse_args()
    diagrams_dir = args.dir
    
    if not os.path.exists(diagrams_dir):
        print(f"Error: Directory {diagrams_dir} does not exist.")
        return 1
    
    if args.check or (not args.check and not args.update and not args.all):
        out_of_sync = check_diagram_sync(diagrams_dir)
        if out_of_sync:
            print(f"\nFound {len(out_of_sync)} diagrams that need updating.")
            if not args.update:
                print("Run with --update to update these diagrams.")
        else:
            print("\nAll diagrams are up to date! 🎉")
    
    if args.update or args.all:
        if args.all:
            print("\nUpdating all diagrams...")
            diagram_pairs = find_diagram_pairs(diagrams_dir)
            for mermaid_file, png_file in diagram_pairs:
                generate_png_from_mermaid(mermaid_file, png_file)
        elif args.update:
            out_of_sync = check_diagram_sync(diagrams_dir)
            if out_of_sync:
                print("\nUpdating out-of-sync diagrams...")
                for mermaid_file, png_file in out_of_sync:
                    generate_png_from_mermaid(mermaid_file, png_file)
            else:
                print("\nAll diagrams are already up to date! 🎉")
    
    return 0


if __name__ == "__main__":
    exit(main())