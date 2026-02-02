"""
Fix indentation for sequential experiment blocks
"""

def fix_indentation():
    filepath = 'run_all_kaggle.py'
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern: lines that start with exactly 4 spaces and "if " (but should be 8 spaces)
    # between sequential mode marker and the post-processing section
    
    lines = content.split('\n')
    new_lines = []
    in_sequential_block = False
    
    for i, line in enumerate(lines):
        # Start sequential block
        if '# Sequential mode: Execute experiments one by one' in line:
            in_sequential_block = True
            new_lines.append(line)
            continue
        
        # End sequential block
        if in_sequential_block and '# Generate comprehensive summary report' in line:
            in_sequential_block = False
            new_lines.append(line)
            continue
        
        # Inside sequential block - fix indentation
        if in_sequential_block:
            if line.strip() == '':
                new_lines.append(line)  # Keep empty lines
            elif line.startswith('        '):
                new_lines.append(line)  # Already has 8 spaces, keep it
            elif line.startswith('    '):
                # Has 4 spaces, needs to become 8
                new_lines.append('    ' + line)
            else:
                # No indent or different (shouldn't happen), keep as-is
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print(f"Fixed indentation in sequential experiment block")

if __name__ == "__main__":
    fix_indentation()
