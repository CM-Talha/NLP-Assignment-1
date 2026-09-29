import json
import base64
import os

with open('NLP_Assignment_1_Sarcastic_Urdu_Data.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

os.makedirs('scratch/images', exist_ok=True)

img_count = 0
for i, cell in enumerate(nb['cells']):
    for j, out in enumerate(cell.get('outputs', [])):
        data = out.get('data', {})
        if 'image/png' in data:
            img_count += 1
            img_data = base64.b64decode(data['image/png'])
            filename = f"scratch/images/fig_cell_{i+1}_out_{j+1}.png"
            with open(filename, 'wb') as img_f:
                img_f.write(img_data)
            print(f"Saved {filename} ({len(img_data):,d} bytes)")

# Cell 20 outputs (Master table)
cell_20 = nb['cells'][19] # 0-indexed
print("\n--- CELL 20 OUTPUTS ---")
for out in cell_20.get('outputs', []):
    if 'text/plain' in out.get('data', {}):
        print("".join(out['data']['text/plain']))
    if out.get('output_type') == 'stream':
        print("".join(out.get('text', [])))

# Also check cell 6 (EDA plots) and cell 10, 12, 14, 16, 18
print(f"\nTotal extracted images: {img_count}")
