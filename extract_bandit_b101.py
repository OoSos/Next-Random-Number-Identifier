import json
import os

def load_bandit_results(json_file_path):
    """
    Loads and parses a Bandit JSON output file.

    Args:
        json_file_path (str): The path to the Bandit JSON output file.
    
    Returns:
        dict: Parsed JSON data, or None if an error occurs (e.g., file not found, JSON decode error).
              Error messages are printed to stderr by this function.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_file_path}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading/parsing '{json_file_path}': {e}")
        return None

def process_bandit_results(bandit_data):
    """
    Extracts B101 issues (assert_used) from parsed Bandit data.

    Args:
        bandit_data (dict): Parsed Bandit JSON data. Expected to be the root of the Bandit report.
    
    Returns:
        list: A list of dictionaries, each with 'filename' and 'line_number' for B101 issues.
              Returns an empty list if no B101 issues are found or if the 'results' key is missing/invalid.
    """
    b101_issues = []
    if not bandit_data:
        # This case should ideally be handled by the caller checking the return of load_bandit_results
        return b101_issues

    results = bandit_data.get("results")

    if isinstance(results, list):
        for issue in results:
            if issue.get('test_id') == 'B101':
                filename = issue.get('filename')
                line_number = issue.get('line_number')
                # Ensure both filename and line_number are present
                if filename and line_number is not None:
                    b101_issues.append({'filename': filename, 'line_number': line_number})
    return b101_issues

def display_b101_issues(issues_list, source_file_description="the report"):
    """
    Displays the found B101 issues or a message if none are found.

    Args:
        issues_list (list): A list of issue dictionaries 
                              (e.g., [{'filename': 'path/to/file.py', 'line_number': 10}]).
        source_file_description (str): A descriptive string for the source of the issues,
                                       used in the output messages.
    """
    if not issues_list:
        print(f"No B101 (assert_used) issues found in {source_file_description}.")
    else:
        print(f"Found B101 (assert_used) issues in {source_file_description} at the following locations:")
        for issue in issues_list:
            # Normalize path for consistent display across OS
            normalized_path = os.path.normpath(issue['filename'])
            print(f"  File: {normalized_path}, Line: {issue['line_number']}")

if __name__ == "__main__":
    # Define the path to the Bandit results JSON file.
    # This path is relative to the project root if the script is run from there.
    bandit_results_file_path = "benchmark_results/bandit_scan_results.json"
    
    # For greater robustness, especially if the script's location relative to the
    # JSON file might change, or if it's run from different directories,
    # constructing an absolute path is recommended. Example:
    # script_directory = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(script_directory) # If script is in a subdirectory
    # absolute_bandit_results_file_path = os.path.join(project_root, bandit_results_file_path)
    # current_file_to_process = absolute_bandit_results_file_path
    
    current_file_to_process = bandit_results_file_path

    # 1. Load and parse the Bandit results file
    parsed_bandit_data = load_bandit_results(current_file_to_process)

    # 2. If loading was successful, process the data to find B101 issues
    if parsed_bandit_data:
        extracted_b101_issues = process_bandit_results(parsed_bandit_data)
        
        # 3. Display the found issues or a message if none were found
        display_b101_issues(extracted_b101_issues, f"'{current_file_to_process}'")
    # If parsed_bandit_data is None, load_bandit_results has already printed an error message.
    # No further action is needed in the main block for that error case.
