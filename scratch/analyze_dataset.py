import os
import pandas as pd
import numpy as np

os.makedirs('scratch', exist_ok=True)

df = pd.read_csv('urdu_sarcastic_dataset.csv')
with open('scratch/dataset_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('=== RAW DATASET INFO ===\n')
    f.write(f'Shape: {df.shape}\n')
    f.write(f'Columns: {df.columns.tolist()}\n')
    f.write(f'Missing urdu_text: {df["urdu_text"].isna().sum()}\n')
    f.write(f'Missing is_sarcastic: {df["is_sarcastic"].isna().sum()}\n\n')

    # Unnamed columns investigation
    for col in df.columns:
        if 'Unnamed' in col:
            non_nulls = df[col].dropna()
            f.write(f'{col}: {len(non_nulls)} non-null entries.\n')

    # Filter out missing
    df_clean = df.dropna(subset=['urdu_text', 'is_sarcastic']).copy()
    # Filter valid binary labels
    df_clean = df_clean[df_clean['is_sarcastic'].isin([0, 1, 0.0, 1.0])].copy()
    df_clean['is_sarcastic'] = df_clean['is_sarcastic'].astype(int)

    f.write('\n=== CLEANED DATASET INFO ===\n')
    f.write(f'Cleaned Shape: {df_clean.shape}\n')
    f.write(f'Class counts:\n{df_clean["is_sarcastic"].value_counts().to_string()}\n')
    f.write(f'Class percentages:\n{(df_clean["is_sarcastic"].value_counts(normalize=True)*100).to_string()}\n')

    # Duplicates analysis
    f.write(f'\nDuplicate urdu_text count: {df_clean.duplicated(subset=["urdu_text"]).sum()}\n')
    f.write(f'Exact duplicate rows count: {df_clean.duplicated().sum()}\n')

    # Contradictory labels (same text, different label)
    text_label_counts = df_clean.groupby('urdu_text')['is_sarcastic'].nunique()
    conflicting_texts = text_label_counts[text_label_counts > 1]
    f.write(f'Texts with conflicting labels (same text labeled both 0 and 1): {len(conflicting_texts)}\n')

    # Length stats
    df_clean['char_len'] = df_clean['urdu_text'].astype(str).str.len()
    df_clean['word_count'] = df_clean['urdu_text'].astype(str).str.split().str.len()

    f.write('\n=== TEXT LENGTH STATISTICS ===\n')
    f.write(df_clean[['char_len', 'word_count']].describe().to_string())
    f.write('\n\n=== LENGTH STATS BY CLASS ===\n')
    f.write('Sarcastic (1):\n')
    f.write(df_clean[df_clean['is_sarcastic'] == 1][['char_len', 'word_count']].describe().to_string())
    f.write('\nNon-Sarcastic (0):\n')
    f.write(df_clean[df_clean['is_sarcastic'] == 0][['char_len', 'word_count']].describe().to_string())

    # Sample rows
    f.write('\n\n=== SAMPLE SARCASTIC ROWS ===\n')
    for idx, row in df_clean[df_clean['is_sarcastic'] == 1].head(5).iterrows():
        f.write(f'[{idx}] {row["urdu_text"]}\n')

    f.write('\n=== SAMPLE NON-SARCASTIC ROWS ===\n')
    for idx, row in df_clean[df_clean['is_sarcastic'] == 0].head(5).iterrows():
        f.write(f'[{idx}] {row["urdu_text"]}\n')

print("Analysis script finished.")
