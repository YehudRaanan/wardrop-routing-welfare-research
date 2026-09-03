import os
import sys
import json
import shutil
import datetime

def timestamped_path(path):
    """
    Converts 'results.csv' -> 'results_20260201_231500.csv'
    Enforces Part 2.3: Never Overwrite Results
    """
    base, ext = os.path.splitext(path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}{ext}"

def backup_before_modify(path):
    """
    Creates timestamped backup in ./backups/
    Enforces Part 2.2: Mandatory Backup Before Modification
    """
    if not os.path.exists(path):
        return
    
    backup_dir = "./backups"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    filename = os.path.basename(path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"{filename}.backup_{timestamp}")
    
    shutil.copy2(path, backup_path)
    print(f"Backup created: {backup_path}")

def require_approval(goal, params, expected_output):
    """
    Blocks execution until user approves.
    Enforces Part 1.3: Mandatory Run Permission
    """
    print("\n" + "="*40)
    print("*** APPROVAL REQUIRED ***")
    print("="*40)
    print(f"GOAL: {goal}")
    print(f"PARAMS: {json.dumps(params, indent=2)}")
    print(f"EXPECTED OUTPUT: {expected_output}")
    print("-" * 40)
    
    try:
        response = input("Type 'YES' to approve run: ")
    except EOFError:
        print("\nInput stream closed. Assuming non-interactive run. Aborting.")
        sys.exit(1)

    if response.strip() != "YES":
        print("Run aborted by user.")
        sys.exit(0)
    print("Run approved. Proceeding...\n")

def save_parameter_snapshot(params, output_dir, run_name=None):
    """
    Saves JSON snapshot before each run.
    Enforces Part 2.4: Parameter Snapshot
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"params_snapshot_{run_name}_" if run_name else "params_snapshot_"
    snapshot_path = os.path.join(output_dir, f"{prefix}{timestamp}.json")
    
    with open(snapshot_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Parameter snapshot saved: {snapshot_path}")

def guarded_run(goal, params, expected_output, run_function):
    """
    Combined convenience function.
    """
    require_approval(goal, params, expected_output)
    
    # Handle output path timestamping
    safe_output_path = timestamped_path(expected_output)
    output_dir = os.path.dirname(safe_output_path)
    if not output_dir:
        output_dir = "."
    
    save_parameter_snapshot(params, output_dir)
    
    # Execute the actual logic
    run_function(safe_output_path)