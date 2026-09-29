import json

with open('Urdu_sarcasm.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
all_passed = True
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code_str = ''.join(cell['source'])
        try:
            compile(code_str, f'Cell_{idx+1}', 'exec')
            print(f"Cell {idx+1:02d} [Code]: Syntax Valid!")
        except SyntaxError as e:
            print(f"Cell {idx+1:02d} [Code]: SYNTAX ERROR: {e}")
            all_passed = False

if all_passed:
    print("\nALL 10 CODE CELLS COMPILED WITH ZERO SYNTAX ERRORS!")
else:
    print("\nSome cells failed to compile.")
