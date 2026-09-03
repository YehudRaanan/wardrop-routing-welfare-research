"""
Path Setup for Project Structure
---------------------------------
Sets up Python paths so any runner can import shared modules.

Usage:
    import path_setup  # This sets up all paths automatically
    import simulation_shared as shared
    from project_guardrails import require_approval
    from core.agent_logic import GlobalEMA
"""

import sys
import os

# Get project root (directory containing this file)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add paths for shared modules
paths_to_add = [
    os.path.join(PROJECT_ROOT, "shared_lib"),              # Config modules
    os.path.join(PROJECT_ROOT, "shared_lib", "core"),      # agent_logic.py
    os.path.join(PROJECT_ROOT, "shared_lib", "network"),   # Network generation
    os.path.join(PROJECT_ROOT, "shared_lib", "analysis"),  # Shared analysis scripts
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Helper function to get paths
def get_config_path(filename):
    """Get absolute path to a file in shared_lib/"""
    return os.path.join(PROJECT_ROOT, "shared_lib", filename)

def get_network_path(filename):
    """Get absolute path to a network file in shared_lib/network/"""
    return os.path.join(PROJECT_ROOT, "shared_lib", "network", filename)

def get_project_root():
    """Get absolute path to project root"""
    return PROJECT_ROOT
