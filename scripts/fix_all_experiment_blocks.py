"""
Comprehensive fix for all sequential experiment blocks
"""
import re

def fix_all_experiment_blocks():
    filepath = 'run_all_kaggle.py'
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match and fix:
    # "        if 'xxx' in selected_experiments:\n        if not check_time_budget"
    # Should be:
    # "        if 'xxx' in selected_experiments:\n            if not check_time_budget"
    
    # Fix pattern 1: Double 'if' on separate lines with same indentation
    pattern1 = re.compile(
        r"(        if '[^']+' in selected_experiments:)\n        (if not check_time_budget)",
        re.MULTILINE
    )
    content = pattern1.sub(r"\1\n            \2", content)
    
    # Fix pattern 2: 'with error_context' that should be more indented
    pattern2 = re.compile(
        r"(            if not check_time_budget\([^)]+\):\n                return experiment_results)\n        (with error_context\()",
        re.MULTILINE
    )
    content = pattern2.sub(r"\1\n            \2", content)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed all experiment block indentations")

if __name__ == "__main__":
    fix_all_experiment_blocks()
