import json
import sys

def patch_notebook(filepath, is_master=False):
    print(f"Patching {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    patched_count = 0
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = cell['source']
        new_source = []
        cell_modified = False

        for line in source:
            # 1. Fix shallow copy bug for BiLSTM
            if "best_model_state = model.state_dict().copy()" in line:
                indent = line[:len(line) - len(line.lstrip())]
                new_line = indent + "best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}\n"
                new_source.append(new_line)
                cell_modified = True
                patched_count += 1
                print(f"  [Cell {idx+1}] Fixed shallow copy bug in BiLSTM checkpointing.")
            # 2. Fix shallow copy bug for Transformer
            elif "best_state = model.state_dict().copy()" in line:
                indent = line[:len(line) - len(line.lstrip())]
                new_line = indent + "best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}\n"
                new_source.append(new_line)
                cell_modified = True
                patched_count += 1
                print(f"  [Cell {idx+1}] Fixed shallow copy bug in Transformer checkpointing.")
            # 3. Fix ablation max_features confounding variable (10000 -> 12000)
            elif "vec_emoji_ablation = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)" in line:
                new_line = line.replace("max_features=10000", "max_features=12000")
                new_source.append(new_line)
                cell_modified = True
                patched_count += 1
                print(f"  [Cell {idx+1}] Standardized Emoji Ablation vectorizer max_features to 12000.")
            elif "raw_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)" in line:
                new_line = line.replace("max_features=10000", "max_features=12000")
                new_source.append(new_line)
                cell_modified = True
                patched_count += 1
                print(f"  [Cell {idx+1}] Standardized Raw Text Ablation vectorizer max_features to 12000.")
            else:
                new_source.append(line)

        if cell_modified:
            cell['source'] = new_source

    indent_size = 1 if is_master else 2
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=indent_size, ensure_ascii=False)

    print(f"Completed {filepath}: {patched_count} patches applied.\n")

if __name__ == '__main__':
    patch_notebook('Urdu_sarcasm.ipynb', is_master=False)
    patch_notebook('NLP_Assignment_1_Sarcastic_Urdu_Data.ipynb', is_master=True)
