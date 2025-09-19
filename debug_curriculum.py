#!/usr/bin/env python3

# Debug curriculum parsing
import re

def debug_parsing():
    import re
    
    with open('attached_assets/output_1758321057482.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    print("Debugging first 20 lines with NEW parsing logic...")
    for i, line in enumerate(lines[:20]):
        if not line.strip():
            continue
            
        print(f"Line {i+1}: '{line}'")
        
        # Clean the line by removing tree characters and getting the content
        clean_line = line.strip()
        # Remove tree drawing characters
        clean_line = re.sub(r'^[│├└\s─]*', '', clean_line).strip()
        print(f"  Clean line: '{clean_line}'")
        
        if not clean_line or clean_line == '.':
            print("  >>> SKIPPED (empty or dot)")
            continue
        
        # Count tree depth by counting the tree structure
        # Pattern is: │ + two non-breaking spaces + one regular space
        tree_pattern = '│\xa0\xa0 '  # │ + two non-breaking spaces (Unicode 160) + one regular space
        tree_depth = line.count(tree_pattern) + (1 if ('├──' in line or '└──' in line) else 0)
        
        print(f"  Tree depth: {tree_depth}")
        
        # Check if it's a level
        if tree_depth == 1 and any(level in clean_line.lower() for level in ['foundation', 'intermediate', 'final']):
            print(f"  >>> DETECTED AS LEVEL: {clean_line}")
        elif tree_depth == 2 and 'paper' in clean_line.lower():
            print(f"  >>> DETECTED AS PAPER: {clean_line}")
        elif tree_depth == 3 and ('module' in clean_line.lower() or 'chapter' in clean_line.lower()):
            print(f"  >>> DETECTED AS MODULE/CHAPTER: {clean_line}")
        
        print("---")

if __name__ == "__main__":
    debug_parsing()