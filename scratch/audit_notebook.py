import json

with open('NLP_Assignment_1_Sarcastic_Urdu_Data.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch/notebook_audit.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        out.write(f"\n{'='*70}\n")
        out.write(f"CELL {i+1}: Type = {cell['cell_type']}\n")
        out.write(f"{'='*70}\n")
        
        src = "".join(cell.get('source', []))
        out.write(f"--- SOURCE (first 1000 chars) ---\n{src[:1000]}\n")
        
        outputs = cell.get('outputs', [])
        if outputs:
            out.write(f"\n--- OUTPUTS ({len(outputs)} items) ---\n")
            for j, o in enumerate(outputs):
                out.write(f"  [Output {j+1}: {o.get('output_type')}]\n")
                if o.get('output_type') == 'stream':
                    out.write(f"    {''.join(o.get('text', []))[:1000]}\n")
                elif o.get('output_type') in ['display_data', 'execute_result']:
                    data = o.get('data', {})
                    for k in data.keys():
                        if k == 'text/plain':
                            out.write(f"    text/plain: {''.join(data[k])[:500]}\n")
                        elif k == 'image/png':
                            out.write(f"    image/png: (length {len(data[k])})\n")
                elif o.get('output_type') == 'error':
                    out.write(f"    ERROR: {o.get('ename')}: {o.get('evalue')}\n")
                    out.write(f"    TRACEBACK:\n    " + "\n    ".join(o.get('traceback', [])[:5]) + "\n")

print("Wrote scratch/notebook_audit.txt successfully!")
