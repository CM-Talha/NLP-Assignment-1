import json

with open('Urdu_sarcasm.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Valid JSON Notebook!')
print('Number of cells:', len(nb['cells']))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
print(f'Code cells: {len(code_cells)}, Markdown cells: {len(md_cells)}')

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    first_line = src.strip().split('\n')[0] if src.strip() else '(empty)'
    cell_type = c['cell_type']
    print(f'Cell {i+1:02d} [{cell_type}]: {first_line[:65]}')
