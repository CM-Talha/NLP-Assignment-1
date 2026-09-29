import json
import os

with open('NLP_Assignment_1_Sarcastic_Urdu_Data.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

with open('scratch/nb_outputs_summary.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        ctype = cell.get('cell_type', '')
        source = "".join(cell.get('source', []))
        first_line = source.strip().split('\n')[0] if source.strip() else '(empty)'
        out.write(f"\n=========================================\n")
        out.write(f"CELL {i+1} [{ctype}]: {first_line[:80]}\n")
        out.write(f"=========================================\n")
        
        if ctype == 'markdown':
            out.write("--- MARKDOWN CONTENT ---\n")
            out.write(source[:500] + ("\n...(truncated)" if len(source) > 500 else "") + "\n")
        elif ctype == 'code':
            outputs = cell.get('outputs', [])
            out.write(f"Outputs count: {len(outputs)}\n")
            for j, o in enumerate(outputs):
                otype = o.get('output_type', '')
                out.write(f"  Output {j+1} [{otype}]:\n")
                if otype == 'stream':
                    text = "".join(o.get('text', []))
                    out.write(text[:1000] + ("\n  ...(stream truncated)" if len(text) > 1000 else "") + "\n")
                elif otype == 'display_data' or otype == 'execute_result':
                    data = o.get('data', {})
                    out.write(f"    Data keys: {list(data.keys())}\n")
                    if 'text/plain' in data:
                        tplain = "".join(data['text/plain'])
                        out.write(f"    text/plain: {tplain[:500]}\n")
                    if 'image/png' in data:
                        out.write(f"    [IMAGE PNG present, base64 length: {len(data['image/png'])}]\n")
                elif otype == 'error':
                    out.write(f"    ERROR: {o.get('ename')}: {o.get('evalue')}\n")

print("Saved output summary to scratch/nb_outputs_summary.txt")
