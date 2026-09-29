import json

with open('Urdu_sarcasm.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch/debug_cells.txt', 'w', encoding='utf-8') as f:
    f.write('--- CELL 7 CODE (lines 95 to 115) ---\n')
    lines7 = ''.join(nb['cells'][6]['source']).split('\n')
    for i, l in enumerate(lines7, 1):
        if 95 <= i <= 118:
            f.write(f'{i:03d}: {repr(l)}\n')

    f.write('\n--- CELL 11 CODE (lines 45 to 65) ---\n')
    lines11 = ''.join(nb['cells'][10]['source']).split('\n')
    for i, l in enumerate(lines11, 1):
        if 45 <= i <= 65:
            f.write(f'{i:03d}: {repr(l)}\n')

print("Written to scratch/debug_cells.txt")
