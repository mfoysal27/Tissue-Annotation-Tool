import ast
import sys

try:
    with open('tissue_segmentation_tool.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to parse the file
    ast.parse(content)
    print("✓ No syntax errors found! The file is syntactically correct.")
    
except SyntaxError as e:
    print(f"✗ Syntax error found:")
    print(f"  Line {e.lineno}: {e.text.strip() if e.text else 'N/A'}")
    print(f"  Error: {e.msg}")
    print(f"  Position: {' ' * (e.offset-1 if e.offset else 0)}^")
    sys.exit(1)
except Exception as e:
    print(f"✗ Other error: {e}")
    sys.exit(1) 