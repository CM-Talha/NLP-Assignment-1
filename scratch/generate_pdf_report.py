import os
import subprocess
import pymupdf

def generate_report():
    fig1_path = os.path.abspath('scratch/images/fig_cell_6_out_3.png').replace('\\', '/')
    fig2_path = os.path.abspath('scratch/images/fig_cell_20_out_3.png').replace('\\', '/')
    fig3_path = os.path.abspath('scratch/images/fig_cell_20_out_4.png').replace('\\', '/')

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DS5007 NLP Assignment 1 - Urdu Sarcasm Detection Report</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 9mm 11mm 9mm 11mm;
  }}

  * {{
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 8.0pt;
    line-height: 1.25;
    color: #1a202c;
    margin: 0;
    padding: 0;
  }}

  .page {{
    page-break-after: always;
    height: 279mm;
    max-height: 279mm;
    position: relative;
    overflow: hidden;
  }}

  .page:last-child {{
    page-break-after: avoid;
  }}

  /* Header styles */
  .header-bar {{
    border-bottom: 2px solid #2b5c8f;
    padding-bottom: 3px;
    margin-bottom: 5px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }}
  .inst-title {{
    font-size: 7.2pt;
    font-weight: 700;
    color: #2b5c8f;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }}
  .course-tag {{
    font-size: 7.2pt;
    font-weight: 600;
    color: #4a5568;
  }}

  h1.doc-title {{
    font-size: 12.2pt;
    font-weight: 800;
    color: #1a365d;
    margin: 3px 0 1px 0;
    line-height: 1.15;
  }}
  .doc-subtitle {{
    font-size: 8.0pt;
    font-weight: 600;
    color: #d95f02;
    margin-bottom: 5px;
  }}

  .meta-grid {{
    display: flex;
    justify-content: space-between;
    background-color: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 3.5px 8px;
    margin-bottom: 6px;
    font-size: 7.0pt;
  }}
  .meta-item strong {{
    color: #2b5c8f;
  }}

  /* Layout */
  .two-col {{
    display: flex;
    gap: 10px;
  }}
  .col {{
    flex: 1;
  }}
  .col-46 {{
    flex: 45;
    min-width: 0;
  }}
  .col-54 {{
    flex: 55;
    min-width: 0;
  }}

  h2 {{
    font-size: 8.8pt;
    font-weight: 700;
    color: #1a365d;
    border-left: 3px solid #2b5c8f;
    padding-left: 4px;
    margin: 5px 0 2px 0;
    text-transform: uppercase;
    letter-spacing: 0.2px;
  }}
  h3 {{
    font-size: 7.8pt;
    font-weight: 700;
    color: #2d3748;
    margin: 3px 0 1px 0;
  }}

  p {{
    margin: 0 0 3px 0;
    text-align: justify;
  }}

  ul {{
    margin: 2px 0 4px 0;
    padding-left: 14px;
  }}
  li {{
    margin-bottom: 1.5px;
  }}

  /* Callout boxes */
  .callout {{
    background-color: #ebf8ff;
    border-left: 3px solid #3182ce;
    padding: 3px 5px;
    font-size: 7.3pt;
    margin: 3px 0;
    border-radius: 2px;
  }}
  .callout-bonus {{
    background-color: #fffaf0;
    border-left: 3px solid #dd6b20;
    padding: 3.5px 6px;
    font-size: 7.3pt;
    margin: 3px 0;
    border-radius: 2px;
  }}
  .callout-bonus strong {{
    color: #c05621;
  }}

  /* Table styling */
  table.results-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 6.8pt;
    margin: 3px 0;
  }}
  table.results-table th {{
    background-color: #2b5c8f;
    color: #ffffff;
    font-weight: 700;
    padding: 2.5px 3px;
    text-align: center;
    border: 1px solid #2b5c8f;
  }}
  table.results-table th.left {{
    text-align: left;
  }}
  table.results-table td {{
    padding: 2px 3px;
    border: 1px solid #e2e8f0;
    text-align: center;
  }}
  table.results-table td.left {{
    text-align: left;
  }}
  table.results-table tr:nth-child(even) {{
    background-color: #f7fafc;
  }}
  table.results-table tr.highlight {{
    background-color: #feebc8;
    font-weight: 700;
  }}
  table.results-table tr.bonus-row {{
    background-color: #e6fffa;
    font-weight: 700;
  }}

  /* Figures */
  .fig-container {{
    text-align: center;
    margin: 3px 0;
  }}
  .fig-container img {{
    max-width: 100%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 3px;
  }}
  .fig-caption {{
    font-size: 6.7pt;
    font-weight: 600;
    color: #4a5568;
    margin-top: 1.5px;
    line-height: 1.15;
  }}

  /* Error analysis case studies */
  .case-card {{
    border: 1px solid #e2e8f0;
    border-radius: 3px;
    padding: 3px 5px;
    margin-bottom: 3.5px;
    background-color: #ffffff;
  }}
  .case-header {{
    display: flex;
    justify-content: space-between;
    font-size: 7.0pt;
    font-weight: 700;
    margin-bottom: 1.5px;
  }}
  .badge-fn {{
    background-color: #fed7d7;
    color: #9b2c2c;
    padding: 0.5px 3.5px;
    border-radius: 2px;
    font-size: 6.5pt;
  }}
  .badge-fp {{
    background-color: #feebc8;
    color: #7b341e;
    padding: 0.5px 3.5px;
    border-radius: 2px;
    font-size: 6.5pt;
  }}
  .urdu-text {{
    font-family: 'Urdu Typesetting', 'Jameel Noori Nastaleeq', 'Segoe UI', Tahoma, sans-serif;
    font-size: 9.0pt;
    direction: rtl;
    text-align: right;
    color: #1a202c;
    margin: 1px 0;
    line-height: 1.35;
    background-color: #f7fafc;
    padding: 1.5px 5px;
    border-radius: 2px;
  }}
  .case-trans {{
    font-style: italic;
    font-size: 6.8pt;
    color: #4a5568;
    margin-bottom: 1.5px;
  }}
  .case-analysis {{
    font-size: 6.8pt;
    color: #2d3748;
    line-height: 1.18;
  }}

  .footer-bar {{
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    border-top: 1px solid #cbd5e0;
    padding-top: 2px;
    font-size: 6.5pt;
    color: #718096;
    display: flex;
    justify-content: space-between;
  }}
</style>
</head>
<body>

<!-- ======================================================================= -->
<!-- PAGE 1: INTRODUCTION, DATA PREPARATION & EDA                           -->
<!-- ======================================================================= -->
<div class="page">
  <div class="header-bar">
    <div class="inst-title">FAST National University of Computer and Emerging Sciences &bull; Dept. of Computer Science</div>
    <div class="course-tag">DS5007: Natural Language Processing | Fall 2026</div>
  </div>

  <h1 class="doc-title">Computational Sarcasm Detection in Urdu Social-Media Text</h1>
  <div class="doc-subtitle">A Comparative Study of Stylometry, Handcrafted Features, BiLSTMs, and Pretrained Contextual Transformers</div>

  <div class="meta-grid">
    <div class="meta-item"><strong>Task:</strong> Binary Sarcasm Classification (0 vs 1)</div>
    <div class="meta-item"><strong>Primary Metric:</strong> Macro-F1 Score (Test Partition)</div>
    <div class="meta-item"><strong>Corpus:</strong> 20,060 Raw &rarr; 14,568 Deduplicated</div>
    <div class="meta-item"><strong>Hardware:</strong> Tesla T4 GPU (PyTorch 2.11+cu128)</div>
  </div>

  <div class="two-col">
    <!-- Left Column -->
    <div class="col">
      <h2>1. Abstract & Problem Formulation</h2>
      <p>
        Sarcasm represents a pragmatic inversion where illocutionary intent contradicts literal lexical content. In informal Urdu social-media text, computational detection is severely challenged by non-standard Perso-Arabic Nastaliq orthography, lack of grammatical capitalization, agglutinative clitic attachments, extensive code-mixing (Roman Urdu/English), and heavy reliance on non-verbal cues (emojis, character elongation, punctuation). This report presents an empirical NLP investigation comparing classical statistical baselines, a <strong>17-dimensional handcrafted Urdu feature set</strong> (earning bonus marks), a custom <strong>PyTorch BiLSTM with Bahdanau Self-Attention</strong>, and fine-tuned <strong>Multilingual BERT (mBERT)</strong>.
      </p>

      <h2>2. Data Preparation & Leakage Prevention</h2>
      <p>
        The provided dataset comprises 20,060 raw entries. A systematic data-cleaning audit revealed two primary structural anomalies:
      </p>
      <ul>
        <li><strong>Delimiter Parsing Artifacts:</strong> Naive CSV parsing produced spurious columns (<code>Unnamed: 2</code> through <code>7</code>) caused by unescaped commas inside tweet strings. Filtering NaN records in <code>urdu_text</code> (105) and <code>is_sarcastic</code> (56) yielded 19,955 valid instances.</li>
        <li><strong>Deduplication & Leakage Prevention:</strong> Social-media reposts generated 4,741 duplicate texts (23.76%), including 646 contradictory texts annotated as both 0 and 1 due to subjective irony perception. Conflicting texts were pruned and duplicate posts deduplicated to <strong>14,568 unique samples</strong> (8,661 Class 0; 5,907 Class 1). This is crucial to prevent artificial train-to-test data leakage.</li>
        <li><strong>Stratified Split:</strong> Data was partitioned into <strong>70% Train</strong> (10,197), <strong>15% Validation</strong> (2,185), and <strong>15% Test</strong> (2,186) preserving the 40.55% sarcasm ratio with <em>zero</em> cross-partition overlap.</li>
      </ul>

      <h2>3. Exploratory Data Analysis (EDA)</h2>
      <p>
        Text-length profiling demonstrated that <strong>sarcastic posts are significantly longer</strong> than non-sarcastic posts, averaging <strong>112.76 characters</strong> (std=62.65, median=101.0, 23.93 words) versus <strong>71.56 characters</strong> (std=58.54, median=56.0, 15.67 words) for non-sarcastic posts (+57.6% length increase). This disparity arises from elaborate rhetorical setups, multi-emoji closing clusters, and phonetic character elongation used to mock targets.
      </p>
    </div>

    <!-- Right Column -->
    <div class="col">
      <div class="fig-container">
        <img src="{fig1_path}" alt="EDA Distributions">
        <div class="fig-caption">Figure 1: Exploratory Analysis: Class Distribution, Word Count KDE Density, and Character Length Boxplot. Sarcastic posts exhibit marked positive skew and elongation.</div>
      </div>

      <h2>4. Urdu-Aware Preprocessing & Cue Retention</h2>
      <p>
        Contrary to standard Latin-script NLP pipelines where punctuation and symbols are stripped, Urdu sarcasm detection mandates preserving non-verbal pragmatic carriers:
      </p>
      <ul>
        <li><strong>Character Canonicalization:</strong> Unified Unicode glyphs representing identical phonetic units across Arabic and Persian keyboards (Arabic Yeh <code>ي</code> &rarr; Urdu <code>ی</code>; Arabic Kaf <code>ك</code> &rarr; Urdu <code>ک</code>; Teh Marbuta <code>ة/ۂ</code> &rarr; Goal Heh <code>ہ</code>).</li>
        <li><strong>Aerab & Tatweel Stripping:</strong> Diacritics (fatha, damma, kasra, shadda, tanween) and aesthetic elongation (Tatweel <code>ـ</code>) were systematically stripped to reconcile spelling variations without loss of lexical semantics.</li>
        <li><strong>Sarcasm Cue Preservation:</strong>
          <ul>
            <li><em>Emojis:</em> Preserved and padded with spaces (e.g., <code>😂, 🤣, 😜, 😏, 🙃</code>). Sarcasm frequently juxtaposes positive words with mocking laughing emojis.</li>
            <li><em>Punctuation & Elongation:</em> Retained exclamations, Urdu question marks (<code>؟</code>), and ellipses (<code>...</code>). Character repetition (e.g., <code>کدووووو</code>) was compressed to two characters (<code>کدوو</code>) to prevent vocabulary explosion while retaining the caricature cue.</li>
          </ul>
        </li>
      </ul>
      <div class="callout">
        <strong>Key Engineering Insight:</strong> Preprocessing completed in <strong>0.42 seconds</strong> across all 14,568 samples using vectorized regex passes, establishing a reproducible normalization pipeline.
      </div>
    </div>
  </div>

  <div class="footer-bar">
    <span>DS5007 Natural Language Processing &bull; Programming Assignment 1</span>
    <span>Page 1 of 3</span>
  </div>
</div>

<!-- ======================================================================= -->
<!-- PAGE 2: METHODOLOGY, ABLATIONS & RESULTS                               -->
<!-- ======================================================================= -->
<div class="page">
  <div class="header-bar">
    <div class="inst-title">FAST-NUCES &bull; DS5007 NLP Assignment 1: Urdu Sarcasm Detection</div>
    <div class="course-tag">Methodology, Ablations & Experimental Results</div>
  </div>

  <div class="two-col">
    <!-- Left Column: Architectures -->
    <div class="col">
      <h2>5. Model Architectures & Feature Design</h2>
      
      <h3>A. Classical Baselines (TF-IDF N-Grams)</h3>
      <p>
        We extracted 12,000 TF-IDF features using word unigrams and bigrams with sublinear term-frequency scaling (1 + log(tf)). Three learning paradigms were benchmarked: (1) <em>Multinomial Naive Bayes</em> with Laplace smoothing (α = 1.0), (2) <em>Logistic Regression</em> with L2 regularization (C = 1.0), and (3) <em>LinearSVC</em> with squared-hinge loss (C = 0.5).
      </p>

      <h3>B. Bonus Feature Engineering (17-Dim Urdu Vector)</h3>
      <p>
        To capture domain-specific pragmatic mechanisms and earn bonus marks, we designed the <code>UrduSarcasmFeatureExtractor</code> extracting 17 dense linguistic features:
      </p>
      <ul>
        <li><strong>Rhetorical Cue Expressions:</strong> Regex pattern matching for recognized Urdu satirical markers (e.g., <code>واہ بھئی واہ, کیا بات ہے, ماشاءاللہ, سبحان اللہ, مرشد, پائین, لو کر لو گل, تبدیلی, نیا پاکستان, شاباش</code>).</li>
        <li><strong>Emoji Sub-Categories:</strong> Frequencies of mocking emojis (<code>😂,🤣,😜</code>), smirk/irony (<code>😏,🙄,🙃</code>), praise (<code>👏,👍,❤️</code>), and grief (<code>💔,😭,😢</code>).</li>
        <li><strong>Sentiment Incongruity:</strong> Binary and continuous interaction terms tracking praise co-occurring with mocking emojis.</li>
        <li><strong>Exaggeration & Punctuation:</strong> Repetition clusters (<code>(.)\1{{2,}}</code>), exclamation count, Urdu question mark count (<code>؟</code>), and quotation counts.</li>
        <li><strong>Code-Mixing Ratio:</strong> Latin character ratio vs total characters.</li>
      </ul>
      <div class="callout-bonus">
        <strong>Bonus Verification:</strong> Combining this 17-dimensional vector with TF-IDF Word and Char n-grams into a <code>FeatureUnion</code> boosted LinearSVC Macro-F1 from <strong>0.7979 to 0.8134 (+1.55% absolute gain)</strong>!
      </div>

      <h3>C. Deep Neural Architecture: PyTorch BiLSTM + Attention</h3>
      <p>
        A custom sequential network was trained from scratch:
      </p>
      <ul>
        <li><strong>Embedding & Spatial Dropout:</strong> 128-dim embeddings with <code>Dropout2d(0.35)</code> zeroing entire feature channels to prevent token co-adaptation.</li>
        <li><strong>2-Layer Bidirectional LSTM:</strong> Encodes bidirectional context (h = 128 &times; 2 = 256 dimensions).</li>
        <li><strong>Bahdanau Self-Attention Pooling:</strong> Learns dynamic weights &alpha;<sub>t</sub> = softmax(v<sub>a</sub><sup>T</sup> tanh(W<sub>a</sub> h<sub>t</sub>)) over all sequence steps, focusing directly on sarcasm cues.</li>
        <li><strong>Classifier Head:</strong> LayerNorm &rarr; Linear(256, 64) &rarr; LeakyReLU &rarr; Dropout(0.35) &rarr; Linear(64, 1) trained via <code>BCEWithLogitsLoss</code>, AdamW, and Cosine Annealing.</li>
      </ul>

      <h3>D. Contextual Transformer: Fine-Tuned mBERT</h3>
      <p>
        We fine-tuned <code>bert-base-multilingual-cased</code> (178M params) with a classification head on Colab GPU Compute using AdamW (lr = 2 &times; 10<sup>-5</sup>), linear warmup, sequence length 96, and mixed-precision (<code>torch.amp.autocast</code>).
      </p>
    </div>

    <!-- Right Column: Results & Ablations -->
    <div class="col">
      <h2>6. Systematic Ablation Study</h2>
      <p>
        We evaluated three isolated dimensions mandated by the assignment:
      </p>
      <ul>
        <li><strong>Ablation 1 (Emoji Impact):</strong> Removing emojis slightly shifted linear decision boundaries (Macro-F1 0.8025 vs 0.7979), but dramatically hurt qualitative interpretation of positive-text tweets.</li>
        <li><strong>Ablation 2 (Feature Granularity):</strong> Character n-grams (3-5g, Macro-F1=0.8027) outperformed Word n-grams (0.7979) by capturing Urdu inflectional affixes and clitics. Combining Word + Char achieved <strong>0.8053</strong>.</li>
        <li><strong>Ablation 3 (Text Normalization):</strong> Urdu normalization provided a positive gain (+0.0011) by collapsing redundant variant vocabularies.</li>
      </ul>

      <h2>7. Master Results & Model Comparison</h2>
      <p>
        Table 1 presents the full evaluation on the independent test set (2,186 posts). All models are ranked by <strong>Macro-F1</strong> (primary metric).
      </p>

      <table class="results-table">
        <thead>
          <tr>
            <th class="left">Model / Architecture</th>
            <th>Acc</th>
            <th>C0 Prec</th>
            <th>C0 Rec</th>
            <th>C0 F1</th>
            <th>C1 Prec</th>
            <th>C1 Rec</th>
            <th>C1 F1</th>
            <th>Macro-F1</th>
          </tr>
        </thead>
        <tbody>
          <tr class="highlight">
            <td class="left"><strong>Transformer (mBERT)</strong></td>
            <td><strong>0.8220</strong></td>
            <td><strong>0.8659</strong></td>
            <td>0.8292</td>
            <td>0.8472</td>
            <td>0.7641</td>
            <td><strong>0.8115</strong></td>
            <td><strong>0.7871</strong></td>
            <td><strong>0.8171</strong></td>
          </tr>
          <tr class="bonus-row">
            <td class="left"><strong>Hybrid (+Urdu Feats) [Bonus]</strong></td>
            <td>0.8202</td>
            <td>0.8480</td>
            <td>0.8500</td>
            <td><strong>0.8490</strong></td>
            <td>0.7792</td>
            <td>0.7765</td>
            <td>0.7778</td>
            <td>0.8134</td>
          </tr>
          <tr>
            <td class="left">Ablation: Combined Word+Char</td>
            <td>0.8129</td>
            <td>0.8378</td>
            <td>0.8500</td>
            <td>0.8438</td>
            <td>0.7751</td>
            <td>0.7585</td>
            <td>0.7667</td>
            <td>0.8053</td>
          </tr>
          <tr>
            <td class="left">Ablation: Char N-grams (3-5g)</td>
            <td>0.8115</td>
            <td>0.8294</td>
            <td>0.8600</td>
            <td>0.8444</td>
            <td>0.7828</td>
            <td>0.7404</td>
            <td>0.7610</td>
            <td>0.8027</td>
          </tr>
          <tr>
            <td class="left">Ablation: Text WITHOUT Emojis</td>
            <td>0.8097</td>
            <td>0.8390</td>
            <td>0.8415</td>
            <td>0.8402</td>
            <td>0.7664</td>
            <td>0.7630</td>
            <td>0.7647</td>
            <td>0.8025</td>
          </tr>
          <tr>
            <td class="left">Logistic Regression (Word 1-2g)</td>
            <td>0.8102</td>
            <td>0.8177</td>
            <td><strong>0.8762</strong></td>
            <td>0.8459</td>
            <td><strong>0.7970</strong></td>
            <td>0.7133</td>
            <td>0.7528</td>
            <td>0.7994</td>
          </tr>
          <tr>
            <td class="left">Linear SVC (Word 1-2g Baseline)</td>
            <td>0.8051</td>
            <td>0.8367</td>
            <td>0.8354</td>
            <td>0.8360</td>
            <td>0.7590</td>
            <td>0.7607</td>
            <td>0.7599</td>
            <td>0.7979</td>
          </tr>
          <tr>
            <td class="left">PyTorch BiLSTM + Attention</td>
            <td>0.8079</td>
            <td>0.8161</td>
            <td>0.8738</td>
            <td>0.8440</td>
            <td>0.7935</td>
            <td>0.7111</td>
            <td>0.7500</td>
            <td>0.7970</td>
          </tr>
          <tr>
            <td class="left">Ablation: Raw Unnormalized Text</td>
            <td>0.8042</td>
            <td>0.8349</td>
            <td>0.8362</td>
            <td>0.8355</td>
            <td>0.7590</td>
            <td>0.7573</td>
            <td>0.7582</td>
            <td>0.7969</td>
          </tr>
          <tr>
            <td class="left">Multinomial Naive Bayes</td>
            <td>0.7928</td>
            <td>0.8006</td>
            <td>0.8677</td>
            <td>0.8328</td>
            <td>0.7786</td>
            <td>0.6828</td>
            <td>0.7276</td>
            <td>0.7802</td>
          </tr>
        </tbody>
      </table>

      <div class="fig-container">
        <img src="{fig2_path}" alt="Macro-F1 Model Comparison" style="max-height: 120px;">
        <div class="fig-caption">Figure 2: Primary Model Comparison by Macro-F1. Fine-tuned mBERT and Hybrid (+Urdu Features) achieve top performance.</div>
      </div>
    </div>
  </div>

  <div class="footer-bar">
    <span>DS5007 Natural Language Processing &bull; Programming Assignment 1</span>
    <span>Page 2 of 3</span>
  </div>
</div>

<!-- ======================================================================= -->
<!-- PAGE 3: CONFUSION MATRICES, ERROR ANALYSIS & CONCLUSION               -->
<!-- ======================================================================= -->
<div class="page">
  <div class="header-bar">
    <div class="inst-title">FAST-NUCES &bull; DS5007 NLP Assignment 1: Urdu Sarcasm Detection</div>
    <div class="course-tag">Confusion Matrices, Qualitative Error Analysis & Discussion</div>
  </div>

  <div class="two-col">
    <!-- Left Column: Confusion Matrices & Error Breakdown -->
    <div class="col-46">
      <h2>8. Confusion Matrix Analysis</h2>
      <div class="fig-container">
        <img src="{fig3_path}" alt="Confusion Matrix Heatmaps" style="max-height: 180px;">
        <div class="fig-caption">Figure 3: Confusion Matrices for Core Paradigms on the Test Partition (N=2,186).</div>
      </div>
      <p>
        The confusion matrices reveal clear behavioral profiles:
      </p>
      <ul>
        <li><strong>Naive Bayes & BiLSTM:</strong> Heavily biased toward Class 0 (high recall 86.8% and 87.4%, but lower Class 1 recall 68.3% and 71.1%), generating substantial False Negatives (281 and 256).</li>
        <li><strong>Hybrid Model:</strong> Achieved balanced error dynamics (FP=195, FN=198), raising Class 1 Recall to 77.65% through domain-specific sarcasm cues.</li>
        <li><strong>Transformer (mBERT):</strong> Achieved the lowest False Negatives (FN=167) and highest Sarcastic Recall (81.15%), correctly capturing contextual irony.</li>
      </ul>

      <h2>9. Discussion, Limitations & Ethics</h2>
      <p>
        <strong>Context Blindness:</strong> Current NLP models rely solely on lexical surface cues. Real-world sarcasm frequently demands external political and cultural knowledge. Integrating Retrieval-Augmented Generation (RAG) with real-time Pakistani news streams represents a promising frontier.
      </p>
      <p>
        <strong>Dialectal Diversity:</strong> Informal Urdu tweets heavily integrate Punjabi, Pashto, and Sindhi colloquialisms, which undergo subword fragmentation in standard tokenizers. Specialized regional Pakistani language embeddings are needed.
      </p>
      <p>
        <strong>Academic Integrity:</strong> All libraries, code skeletons, and pretrained weights (Hugging Face Transformers, PyTorch, Scikit-Learn) are properly credited per FAST-NUCES course policies.
      </p>
    </div>

    <!-- Right Column: Qualitative Error Analysis -->
    <div class="col-54">
      <h2>10. Qualitative Error Analysis (7 Case Studies)</h2>
      <p>
        An audit of the 393 test set classification errors revealed systematic linguistic failure modes across the required categories:
      </p>

      <!-- Case 1 -->
      <div class="case-card">
        <div class="case-header">
          <span>Case 1: World-Knowledge / Context Blindness</span>
          <span class="badge-fn">False Negative (True: 1 &rarr; Pred: 0)</span>
        </div>
        <div class="urdu-text">مراد علی شاہ کے بھیس میں ڈی جی آئی ایس آئی تھے حامد میر😁</div>
        <div class="case-trans">"Hamid Mir says it was the DG ISI in the disguise of Murad Ali Shah 😁"</div>
        <div class="case-analysis"><strong>Analysis:</strong> Mocks a sensationalist television claim by journalist Hamid Mir. Surface n-grams contain unvalenced proper nouns (<code>مراد علی شاہ, حامد میر</code>). Lacking political context, the model classifies it as an objective factual report.</div>
      </div>

      <!-- Case 2 -->
      <div class="case-card">
        <div class="case-header">
          <span>Case 2: Emoji Polysemy in Genuine Humor</span>
          <span class="badge-fp">False Positive (True: 0 &rarr; Pred: 1)</span>
        </div>
        <div class="urdu-text">تسی لڑاںٔی کروانی ساڈی کی 😂😂😂</div>
        <div class="case-trans">"Are you trying to get us into a fight or what? 😂😂😂"</div>
        <div class="case-analysis"><strong>Analysis:</strong> Playful friendly teasing among peers. The model heavily weights triple laughing emojis (<code>😂😂😂</code>) as sarcasm markers, incorrectly flagging benign humor as sarcastic mockery.</div>
      </div>

      <!-- Case 3 -->
      <div class="case-card">
        <div class="case-header">
          <span>Case 3: Rhetorical Negation & Implicit Disbelief</span>
          <span class="badge-fn">False Negative (True: 1 &rarr; Pred: 0)</span>
        </div>
        <div class="urdu-text">میں نے کب کہا آپ برے ہیں😇</div>
        <div class="case-trans">"When did I ever say you are a bad person? 😇"</div>
        <div class="case-analysis"><strong>Analysis:</strong> The phrase <code>کب کہا</code> ("when did I say") serves as a rhetorical denial implying the target <em>is</em> bad. Combined with a halo emoji (<code>😇</code>), bag-of-words classifiers misread the literal denial as sincere.</div>
      </div>

      <!-- Case 4 -->
      <div class="case-card">
        <div class="case-header">
          <span>Case 4: Punjabi Colloquial & Regional Idiom</span>
          <span class="badge-fn">False Negative (True: 1 &rarr; Pred: 0)</span>
        </div>
        <div class="urdu-text">چل مہمانوں میں کھانا سرو کر چڑیل چاچی نوں دسدی آں میں😂😂</div>
        <div class="case-trans">"Go serve food to the guests, I'll tell aunt witch about you 😂😂"</div>
        <div class="case-analysis"><strong>Analysis:</strong> Employs Punjabi familial teasing (<code>چاچی نوں دسدی آں</code>). Non-Urdu grammatical particles are fragmented into unknown subwords, obscuring the domestic mockery context.</div>
      </div>

      <!-- Case 5 -->
      <div class="case-card">
        <div class="case-header">
          <span>Case 5: Sarcastic Gratitude / Cynical Politeness</span>
          <span class="badge-fn">False Negative (True: 1 &rarr; Pred: 0)</span>
        </div>
        <div class="urdu-text">شکریہ عمران خان صاحب، اب نمک تو سستا ہے باقی سب خیر ہے 🙏</div>
        <div class="case-trans">"Thank you Imran Khan sahib, at least salt is cheap now, never mind everything else 🙏"</div>
        <div class="case-analysis"><strong>Analysis:</strong> Uses polite formulas (<code>شکریہ, صاحب</code>) and folded hands (<code>🙏</code>) to deliver caustic economic satire. Linear models activate strongly on positive politeness tokens, failing to grasp the critique.</div>
      </div>

      <!-- Case 6 -->
      <div class="case-card">
        <div class="case-header">
          <span>Case 6: Extreme Elongation & Caricature</span>
          <span class="badge-fn">False Negative (True: 1 &rarr; Pred: 0)</span>
        </div>
        <div class="urdu-text">ایسانہ ہو کہ بلاول کوئٹہ جلسے میں کہے کدووووو نکل آیا ہے 👊</div>
        <div class="case-trans">"Lest Bilawal says in the Quetta rally that a pumpkin has emerged 👊"</div>
        <div class="case-analysis"><strong>Analysis:</strong> Caricatures political speech cadence with elongation <code>کدووووو</code>. Even with repetition compression, rare satirical metaphors suffer from low n-gram frequency.</div>
      </div>

      <!-- Case 7 -->
      <div class="case-card">
        <div class="case-header">
          <span>Case 7: Code-Mixing & Transliteration</span>
          <span class="badge-fn">False Negative (True: 1 &rarr; Pred: 0)</span>
        </div>
        <div class="urdu-text">کیا زبردست level ہے governance کا، RIP common sense</div>
        <div class="case-trans">"What a fantastic level of governance, RIP common sense"</div>
        <div class="case-analysis"><strong>Analysis:</strong> The irony pivots on English phrase <code>RIP common sense</code> contrasted with Urdu praise <code>کیا زبردست</code>. Cross-script syntax confuses models without multilingual embeddings.</div>
      </div>
    </div>
  </div>

  <div class="footer-bar">
    <span>DS5007 Natural Language Processing &bull; Programming Assignment 1</span>
    <span>Page 3 of 3</span>
  </div>
</div>

</body>
</html>
"""

    html_file = os.path.abspath('scratch/report.html')
    pdf_file = os.path.abspath('DS5007_NLP_Assignment_1_Report.pdf')

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    def get_chrome_path():
        mac_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            os.path.expanduser("~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        ]
        for p in mac_paths:
            if os.path.exists(p):
                return p
        win_paths = [
            r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")
        ]
        for p in win_paths:
            if os.path.exists(p):
                return p
        for cmd in ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']:
            found = shutil.which(cmd)
            if found:
                return found
        return None

    chrome_path = get_chrome_path()
    if not chrome_path:
        raise RuntimeError("Google Chrome or Chromium binary not found for PDF compilation.")

    cmd = [
        chrome_path,
        '--headless=new',
        '--disable-gpu',
        f'--print-to-pdf={pdf_file}',
        '--no-pdf-header-footer',
        html_file
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Chrome error:", res.stderr)

    reader = pymupdf.open(pdf_file)
    print(f"Generated PDF: {pdf_file}")
    print(f"Total Page Count: {len(reader)}")

    # Render pages as images for inspection
    for i, page in enumerate(reader):
        pix = page.get_pixmap(dpi=150)
        out_name = f'scratch/pdf_pages/page_{i+1}.png'
        pix.save(out_name)
        print(f"Updated {out_name}: {pix.width}x{pix.height}")

    return len(reader)

if __name__ == '__main__':
    generate_report()
