# Define the template product functions.
# This file dynamically imports all template product functions from the fcns folder.

import os
import sys
import importlib
from collections import defaultdict

def fitting_functions():
    """
    Returns a dictionary of template product functions organized by input variable count.
    
    This function dynamically imports all template product function modules from
    the fcns folder and automatically organizes them by their x_dim variable.
    
    To add a new template product function, create a new .py file in lib/fcns/
    with fcn_*, txt_*, dct_* and x_dim. The function will be automatically included.
    """
    
    # Dynamically import all template product functions from the fcns directory and organize by x_dim
    fcns_dir = os.path.join(os.path.dirname(__file__), 'fcns')
    functions_by_dim = defaultdict(list)
    
    # Import all Python files in the fcns directory (except __init__.py)
    for filename in sorted(os.listdir(fcns_dir)):  # Sort for consistent ordering
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = filename[:-3]  # Remove .py extension
            try:
                module = importlib.import_module(f'.fcns.{module_name}', package='lib')
                
                # Get x_dim from the module
                x_dim = getattr(module, 'x_dim', None)
                if x_dim is None:
                    print(f"Warning: Module {module_name} does not define x_dim, skipping", file=sys.stderr)
                    continue
                
                # Look for dct_* variable in the module
                dct_found = False
                for attr_name in dir(module):
                    if attr_name.startswith('dct_'):
                        dct = getattr(module, attr_name)
                        functions_by_dim[x_dim].append(dct)
                        dct_found = True
                        break  # Only one dct per module
                
                if not dct_found:
                    print(f"Warning: Module {module_name} does not define dct_*, skipping", file=sys.stderr)
                    
            except Exception as e:
                # If a module fails to import, log the error but continue
                # This prevents one broken module from breaking the entire system
                print(f"Warning: Failed to import module {module_name}: {e}", file=sys.stderr)
    
    # Convert defaultdict to regular dict for the return value
    functionDictionary = dict(functions_by_dim)
    
    return functionDictionary
