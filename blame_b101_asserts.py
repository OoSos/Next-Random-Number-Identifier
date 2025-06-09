import subprocess
import os
# Assumes extract_bandit_b101.py is in the same directory or accessible via PYTHONPATH
from extract_bandit_b101 import extract_b101_issues

def get_git_blame_author(repo_root, relative_file_path, line_number):
    """
    Runs git blame on the specified file and line number and returns the author.

    Args:
        repo_root (str): Absolute path to the root of the git repository.
        relative_file_path (str): Path to the file, relative to the repo_root.
        line_number (int): The line number to blame.
    """
    # Ensure the relative_file_path uses the OS-specific separator for git,
    # although git on Windows often handles forward slashes too.
    # os.path.normpath might convert to backslashes on Windows.
    # Git typically expects forward slashes or can handle OS-specific ones.
    # Forcing forward slashes for cross-platform git command consistency might be safer
    # if issues arise, but usually not needed if CWD is correct.
    
    command = [
        "git", "blame",
        "-L", f"{line_number},{line_number}",
        "--porcelain",
        "--",  # To disambiguate file path from other options/revisions
        relative_file_path
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,  # Will raise CalledProcessError for non-zero exit codes
            cwd=repo_root,  # Execute in the context of the git repository
            encoding='utf-8',
            errors='replace' # Handle potential encoding errors in git output
        )
        output = process.stdout
        for line in output.splitlines():
            if line.startswith("author "):
                return line.split(" ", 1)[1]
        return "Author not found in blame output (porcelain format issue)"
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.strip() if e.stderr else e.stdout.strip()
        if not error_output: # if stderr/stdout are empty, use a generic message
             error_output = "No specific error message from git."

        if "no such path" in error_output.lower() or "does not exist" in error_output.lower():
            return f"File not found in repository: {relative_file_path} (Error: {error_output})"
        if "fatal: file" in error_output.lower() and "has only" in error_output.lower() and "lines" in error_output.lower():
            return f"Line {line_number} out of range for file {relative_file_path} (Error: {error_output})"
        if "not a git repository" in error_output.lower():
            return f"Not a git repository or .git directory not found at {repo_root}. (Error: {error_output})"
        if "ambiguous argument" in error_output.lower() and "unknown revision or path not in the working tree" in error_output.lower():
            return f"Pathspec '{relative_file_path}' ambiguous or not in working tree. (Error: {error_output})"
        
        return f"git blame failed for {relative_file_path}:{line_number}. Git error: {error_output}"
    except FileNotFoundError:
        return "git command not found. Ensure git is installed and in your system's PATH."
    except Exception as e:
        return f"An unexpected error occurred during git blame: {str(e)}"

def main():
    workspace_root = r"c:\Users\Owner\GitHubProjects\Next-Random-Number-Identifier"
    bandit_report_relative_path = "benchmark_results/bandit_scan_results.json"
    absolute_bandit_report_path = os.path.join(workspace_root, bandit_report_relative_path)

    print(f"Attempting to read Bandit report from: {absolute_bandit_report_path}")
    issues = extract_b101_issues(absolute_bandit_report_path)

    if issues is None:
        print("Error: Could not retrieve B101 issues (problem reading or parsing Bandit report).")
        return
    if not issues:
        print("No B101 issues found in the report to blame.")
        return

    print(f"\nFound {len(issues)} B101 (assert_used) issues. Blaming authors (skipping venv)...")
    count_processed = 0
    count_skipped_venv = 0
    count_skipped_other = 0
    
    for issue in issues:
        relative_file_path = issue['filename']
        line_number = issue['line_number']

        normalized_relative_path = os.path.normpath(relative_file_path)

        # Filter out files located within a 'venv' directory
        # This checks if the path starts with 'venv' followed by the OS separator.
        if normalized_relative_path.startswith("venv" + os.sep):
            count_skipped_venv += 1
            continue
        
        if not normalized_relative_path.lower().endswith(".py"):
            count_skipped_other += 1
            continue
        
        count_processed += 1
        print(f"  Processing ({count_processed}/{len(issues) - count_skipped_venv - count_skipped_other} relevant): File: {normalized_relative_path}, Line: {line_number}")
        
        author = get_git_blame_author(workspace_root, normalized_relative_path, line_number)
        print(f"    -> Author: {author}")

    print(f"\nBlame process complete.")
    print(f"Total issues from report: {len(issues)}")
    print(f"Issues processed for blame: {count_processed}")
    print(f"Issues skipped (in venv): {count_skipped_venv}")
    print(f"Issues skipped (not .py): {count_skipped_other}")

if __name__ == "__main__":
    main()
