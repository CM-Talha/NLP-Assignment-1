import json
import sys

with open('NLP_Assignment_1_Sarcastic_Urdu_Data.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

steps = []

for idx, cell in enumerate(nb['cells']):
    ctype = cell.get('cell_type')
    src = "".join(cell.get('source', []))
    outs = cell.get('outputs', [])
    
    # Identify section title
    if ctype == 'markdown' and ('# Section' in src or '# DS5007' in src):
        title_line = [l for l in src.splitlines() if l.startswith('#')][0]
        steps.append({
            'type': 'section',
            'index': idx + 1,
            'title': title_line
        })
    elif ctype == 'code':
        stream_text = ""
        data_text = ""
        for o in outs:
            if o.get('output_type') == 'stream':
                stream_text += "".join(o.get('text', []))
            elif o.get('output_type') in ['display_data', 'execute_result']:
                data = o.get('data', {})
                if 'text/plain' in data:
                    data_text += "".join(data['text/plain']) + "\n"
        steps.append({
            'type': 'code',
            'index': idx + 1,
            'lines_of_code': len(src.splitlines()),
            'stream': stream_text,
            'data': data_text
        })

with open('scratch/step_by_step_execution_report.txt', 'w', encoding='utf-8') as out:
    out.write("=" * 80 + "\n")
    out.write("COMPREHENSIVE STEP-BY-STEP EXECUTION & PERFORMANCE AUDIT\n")
    out.write("=" * 80 + "\n\n")
    for s in steps:
        if s['type'] == 'section':
            out.write(f"\n>>> {s['title']} (Cell {s['index']}) <<<\n\n")
        elif s['type'] == 'code':
            out.write(f"--- Code Cell {s['index']} ({s['lines_of_code']} lines) ---\n")
            if s['stream']:
                out.write("STDOUT / LOGS:\n" + s['stream'] + "\n")
            if s['data']:
                out.write("DATA / TABLES / PLOTS:\n" + s['data'] + "\n")

print("Generated scratch/step_by_step_execution_report.txt successfully!")
