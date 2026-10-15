# Project Analysis & Architecture Memory: NLP Urdu Sarcasm Detection (DS5007)

## 1. System Overview
Graduate-level NLP benchmark for Urdu binary sarcasm classification (Sarcastic=1 vs Non-Sarcastic=0) on ~20,000 informal social media posts.

## 2. Core Architectural Components & Verified Authenticity
- Data Ingestion & Sanitization: 3-tier loader handling raw CSV format aberrations (trailing columns, unescaped commas in Urdu text, 597 conflicting label duplicates). Verified mathematically and functionally.
- Urdu-Aware Preprocessing: `UrduPreprocessor` normalizes Unicode variations (Yeh, Kaf, Heh), strips diacritics (Aerab), compresses character elongations (>=3 -> 2), and pads emojis and punctuation with spaces to preserve affective and stylistic sarcasm cues.
- Baseline Models: TF-IDF (1-2 word n-grams) + MultinomialNB (0.7802 F1), LogisticRegression (0.7994 F1), LinearSVC (0.7979 F1). Verified on deduplicated split.
- Bonus Feature Engineering: `UrduSarcasmFeatureExtractor` extracting a 17-dimensional vector (discourse cues, emoji polarity, sentiment incongruity, elongation, punctuation intensity, code-mixing) yielding +1.55% Macro-F1 improvement (0.8134 Macro-F1) via scikit-learn FeatureUnion.
- Deep Neural Model: PyTorch `UrduBiLSTMAttentionClassifier` with spatial dropout, 2-layer BiLSTM, and Bahdanau self-attention pooling (0.7970 Macro-F1). Masking verified.
- Contextual Transformer: Fine-tuned `bert-base-multilingual-cased` achieving SOTA (0.8171 Macro-F1, 82.20% accuracy). Evaluated with mixed precision and linear warmup.
- Ablations & Error Analysis: Systematic 3-part ablation study and 12-case qualitative error study.

## 3. Discovered Vulnerabilities & Applied Patches
- PyTorch State Persistence: Patched shallow `state_dict().copy()` in both notebooks and generators with deep clone `{k: v.cpu().clone() for k, v in model.state_dict().items()}` to ensure true restoration of optimal validation checkpoints.
- Confounding Variables in Ablations: Standardized `max_features=12000` across all ablation conditions in both notebooks.
- Colab MCP Bridge: Fixed hardcoded Windows path with cross-platform Chrome binary resolution and graceful fallback to `webbrowser`.
- Repository Metadata: Created root `README.md`, `requirements.txt`, and `.gitignore`.
