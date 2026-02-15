"""
Final comprehensive fix for sequential experiment blocks indentation
"""

def final_fix():
    filepath = 'run_all_kaggle.py'
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find sequential block boundaries
    sequential_start = None
    sequential_end = None
    
    for i, line in enumerate(lines):
        if '# Sequential mode: Execute experiments one by one' in line:
            sequential_start = i
        if sequential_start and '# Generate comprehensive summary report' in line:
            sequential_end = i
            break
    
    if not sequential_start or not sequential_end:
        print("ERROR: Could not find sequential block")
        return
    
    print(f"Processing lines {sequential_start} to {sequential_end}")
    
    # Fix indentation in sequential block
    i = sequential_start
    while i < sequential_end:
        line = lines[i]
        
        # Skip first few setup lines
        if i <= sequential_start + 2:
            i += 1
            continue
        
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        
        # If line starts with 4 spaces (base indent), should be 8 spaces
        if current_indent == 4 and stripped:
            lines[i] = '    ' + line
        # If line starts with 8 spaces and is content inside with/if blocks, should be 12
        elif current_indent == 8 and not stripped.startswith(('if ', 'with ')):
            lines[i] = '    ' + line
        
        i += 1
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Fixed all indentations in sequential block")

if __name__ == "__main__":
    final_fix()
