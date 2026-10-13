import json

for fname in ['Urdu_sarcasm.ipynb', 'NLP_Assignment_1_Sarcastic_Urdu_Data.ipynb']:
    print('==============================')
    print('FILE:', fname)
    print('==============================')
    with open(fname, encoding='utf-8') as f:
        nb = json.load(f)
    for i, c in enumerate(nb['cells']):
        t = c['cell_type']
        src = ''.join(c['source']).strip()
        lines = src.split('\n')
        first = lines[0] if lines else ''
        out_cnt = len(c.get('outputs', []))
        print(f"Cell {i+1} [{t:8s}] (out: {out_cnt:2d}, lines: {len(lines):3d}): {first[:70]}")
