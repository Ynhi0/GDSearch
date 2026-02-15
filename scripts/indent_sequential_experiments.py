"""
Script to add proper indentation to sequential experiment blocks
"""
import re

def indent_sequential_experiments():
    filepath = 'run_all_kaggle.py'
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the line number where we need to start indenting
    # Look for: "    if 'mnist' in selected_experiments:"
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if "# Sequential mode: Execute experiments one by one" in line:
            start_idx = i + 2  # Start indenting from the line after blank line
        elif start_idx is not None and "# Generate comprehensive summary report" in line:
            end_idx = i
            break
    
    if start_idx is None or end_idx is None:
        print(f"ERROR: Could not find boundaries (start={start_idx}, end={end_idx})")
        return False
    
    print(f"Indenting lines {start_idx+1} to {end_idx}")
    
    # Indent the lines in that range
    for i in range(start_idx, end_idx):
        line = lines[i]
        if line.strip():  # Non-empty line
            if not line.startswith('        '):  # Not already indented to 8 spaces
                lines[i] = '    ' + line
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("SUCCESS: Added indentation to sequential experiment blocks")
    return True

if __name__ == "__main__":
    indent_sequential_experiments()
