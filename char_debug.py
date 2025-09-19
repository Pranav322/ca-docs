#!/usr/bin/env python3

# Debug the exact characters in the tree structure
def char_debug():
    with open('attached_assets/output_1758321057482.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i in range(2, 6):  # Check lines 3-6 (index 2-5)
        line = lines[i].rstrip('\n')
        print(f"Line {i+1}: '{line}'")
        print(f"  Chars: {[c for c in line]}")
        print(f"  Ord values: {[ord(c) for c in line]}")
        
        # Test count patterns
        print(f"  Count '│   ': {line.count('│   ')}")
        print(f"  Count '├──': {line.count('├──')}")
        print(f"  Count '└──': {line.count('└──')}")
        
        tree_depth = line.count('│   ') + (1 if ('├──' in line or '└──' in line) else 0)
        print(f"  Calculated tree depth: {tree_depth}")
        print("---")

if __name__ == "__main__":
    char_debug()