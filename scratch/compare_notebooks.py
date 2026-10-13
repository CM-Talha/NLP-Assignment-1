import json

nb1 = json.load(open('NLP_Assignment_1_Sarcastic_Urdu_Data.ipynb', encoding='utf-8'))
nb2 = json.load(open('Urdu_sarcasm.ipynb', encoding='utf-8'))

print(f"nb1 (NLP_Assignment_1) cells: {len(nb1['cells'])}")
print(f"nb2 (Urdu_sarcasm) cells: {len(nb2['cells'])}")

for i in range(max(len(nb1['cells']), len(nb2['cells']))):
    c1 = nb1['cells'][i] if i < len(nb1['cells']) else None
    c2 = nb2['cells'][i] if i < len(nb2['cells']) else None
    
    t1 = c1['cell_type'] if c1 else 'NONE'
    t2 = c2['cell_type'] if c2 else 'NONE'
    
    s1 = "".join(c1['source']) if c1 else ""
    s2 = "".join(c2['source']) if c2 else ""
    
    first1 = s1.splitlines()[0][:40] if s1 else ""
    first2 = s2.splitlines()[0][:40] if s2 else ""
    
    match = (s1 == s2)
    print(f"Cell {i+1:2d} | match: {str(match):5s} | nb1: [{t1:8s}] {first1:40s} | nb2: [{t2:8s}] {first2:40s}")
