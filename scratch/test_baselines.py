import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Load and clean data
df = pd.read_csv('urdu_sarcastic_dataset.csv')
df = df.dropna(subset=['urdu_text', 'is_sarcastic']).copy()
df = df[df['is_sarcastic'].isin([0, 1, 0.0, 1.0])].copy()
df['is_sarcastic'] = df['is_sarcastic'].astype(int)

# Deduplicate to prevent test leakage
# If duplicate texts exist with conflicting labels, drop or keep first
df = df.drop_duplicates(subset=['urdu_text'], keep='first').copy()
print(f"Unique samples after deduplication: {len(df)}")

X = df['urdu_text']
y = df['is_sarcastic']

# Stratified split: 70% train, 15% val, 15% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.17647, random_state=42, stratify=y_train_val
) # 0.85 * 0.17647 ~= 0.15

print(f"Train size: {len(X_train)} ({len(X_train)/len(df):.1%})")
print(f"Val size: {len(X_val)} ({len(X_val)/len(df):.1%})")
print(f"Test size: {len(X_test)} ({len(X_test)/len(df):.1%})")

# Baseline: TF-IDF Word
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
pred_nb = nb.predict(X_test_vec)
print("\n--- MultinomialNB (Word 1-2 gram) ---")
print(f"Accuracy: {accuracy_score(y_test, pred_nb):.4f}")
print(f"Macro F1: {f1_score(y_test, pred_nb, average='macro'):.4f}")

lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_train_vec, y_train)
pred_lr = lr.predict(X_test_vec)
print("\n--- LogisticRegression (Word 1-2 gram) ---")
print(f"Accuracy: {accuracy_score(y_test, pred_lr):.4f}")
print(f"Macro F1: {f1_score(y_test, pred_lr, average='macro'):.4f}")

svm = LinearSVC(C=0.5, random_state=42)
svm.fit(X_train_vec, y_train)
pred_svm = svm.predict(X_test_vec)
print("\n--- LinearSVC (Word 1-2 gram) ---")
print(f"Accuracy: {accuracy_score(y_test, pred_svm):.4f}")
print(f"Macro F1: {f1_score(y_test, pred_svm, average='macro'):.4f}")

# Character n-grams
char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=15000)
X_train_char = char_vec.fit_transform(X_train)
X_test_char = char_vec.transform(X_test)

svm_char = LinearSVC(C=0.5, random_state=42)
svm_char.fit(X_train_char, y_train)
pred_svm_char = svm_char.predict(X_test_char)
print("\n--- LinearSVC (Char_wb 3-5 gram) ---")
print(f"Accuracy: {accuracy_score(y_test, pred_svm_char):.4f}")
print(f"Macro F1: {f1_score(y_test, pred_svm_char, average='macro'):.4f}")
