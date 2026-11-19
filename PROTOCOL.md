# Reproducibility & Stress-Test Protocol

**Target Paper:** Lee et al. (2021). *Pushing on Text Readability Assessment: A Transformer Meets Handcrafted Linguistic Features* (EMNLP 2021).  
**Author:** [Your Name]  
**Hardware:** Local NVIDIA GeForce RTX 2060 (6GB VRAM) / Google Colab T4 (16GB VRAM)  
**Protocol Version:** `v1.1` (Updated with dual-dataset scope, leakage audit, and feature family ablations)

---

## 1. Objective & Scope
Independently re-implement the hybrid `RoBERTa + Linguistic Features` architecture in PyTorch (without using the authors' original codebase) and test whether its reported gains over a text-only Transformer hold up under:
1. **Multi-seed variance** (testing stability across 5 random seeds).
2. **Data leakage controls** (comparing standard splits against topic-grouped splits).
3. **Domain & scale shifts** (testing on both a small discrete dataset and a larger continuous dataset).
4. **Data scarcity & low-compute setups** (simulating real-world school constraints).

---

## 2. Datasets & Preprocessing Rules
To ensure results are not tied to the quirks of a single small corpus, we evaluate across two standard readability benchmarks:

### 2.1 OneStopEnglish (Classification — 3 Levels)
- **Size:** 567 texts (189 news articles rewritten into *Elementary*, *Intermediate*, and *Advanced* levels).
- **Cleaning Rule:** Strip leaked difficulty headers (e.g., `"Elementary"`, `"Intermediate"`, `"Advanced"`) from the first two lines of every file via regex so the model cannot cheat using metadata.
- **Dual-Split Audit:** Because each article appears 3 times, standard splitting leaks topics between train and test sets. We run both:
  - `StratifiedKFold (k=5)`: Replicates standard paper conditions.
  - `GroupKFold (k=5)`: Groups by `article_id` to prevent topic leakage and measure its true impact on accuracy.

### 2.2 CLEAR Corpus (Regression — Continuous Score)
- **Size:** ~4,724 educational text excerpts rated via Bradley-Terry pairwise comparisons (`BT_easiness`).
- **Splitting Rule:** `StratifiedKFold (k=5)` binned across 5 difficulty quantiles (`pd.qcut`) so every fold shares the same readability distribution.
- **Error Analysis:** Uses human rater standard error (`BT_s_e`) to check whether model errors happen on texts where human raters also disagreed.

---

## 3. Controlled Deviations & Architecture Choices
To keep the pipeline clean, self-contained, and deterministic, the following design choices are fixed:
- **Deterministic Linguistic Features ($d = 22$):** Instead of the 255 external features in the original paper, we extract 22 transparent, reproducible features using Python and `spaCy` (`en_core_web_sm`), grouped into three clear families:
  1. *Surface (8):* char/word/sentence counts, avg word/sentence length, syllables per word, long-word ratio, and Flesch-Kincaid Grade Level.
  2. *Lexical Diversity (4):* TTR, Root-TTR, Corrected-TTR, and hapax legomena ratio.
  3. *Syntactic / POS (10):* POS tag ratios (`NOUN`, `VERB`, `ADJ`, `ADV`, `PRON`, `ADP`, `CCONJ`/`SCONJ`), max/mean dependency tree depth, and subordinate clause count per sentence.
- **Leak-Free Scaling:** `StandardScaler` is fit **only** on the training folds at each split and then applied to validation/test folds.
- **Training & VRAM Setup:** Effective `batch_size = 16` (`physical batch_size = 8` + `gradient_accumulation_steps = 2`) with `fp16` mixed precision to fit within 6GB VRAM. Optimizer: `AdamW` (`lr = 2e-5`, `weight_decay = 0.01`, linear warmup over 10% of total steps, 5 epochs).

---

## 4. Experimental Matrix

### Experiment A: Core Replication & Leakage Audit (100% Data)
- **Models:**
  - `M1 (Baseline):` `roberta-base` + Linear Head.
  - `M2 (Hybrid):` `roberta-base` + All 22 Linguistic Features + LayerNorm + Linear Head.
- **Runs:** 5 Folds $\times$ 5 Seeds (`42, 100, 2024, 777, 999`) = 25 runs per setup across OneStopEnglish (`Stratified` & `Grouped`) and CLEAR.
- **Goal:** Check if `M2 > M1` is statistically significant ($p < 0.05$, paired t-test) and quantify how much accuracy drops when topic leakage is removed.

### Experiment B: Linguistic Feature Family Ablation
- **Condition:** Train `M2` using only one feature family at a time (*Surface-only*, *Lexical-only*, *Syntactic-only*) vs. *All features*.
- **Runs:** 5 Folds $\times$ 3 Seeds (`42, 100, 2024`).
- **Goal:** Identify which linguistic signals actually add value to `RoBERTa` and which ones act as redundant noise.

### Experiment C: Low-Resource Data Scarcity Test
- **Condition:** Subsample training folds to **25%** and **50%** (keeping test folds at 100%).
- **Runs:** Compare `M1` vs. `M2` across 5 Folds $\times$ 3 Seeds.
- **Goal:** Test whether handcrafted features help prevent overfitting when labeled training data is scarce.

### Experiment D: Low-Compute Educational Deployment
- **Condition:** Swap `roberta-base` (125M params) for `distilbert-base-uncased` (66M params).
- **Runs:** Compare `DistilBERT` vs. `DistilBERT + Features` across 5 Folds $\times$ 3 Seeds.
- **Goal:** Measure inference speed, peak VRAM usage, and whether linguistic features recover accuracy lost by model compression.

---

## 5. Evaluation Metrics & Logging
All experiments report **Mean $\pm$ Standard Deviation** across folds and seeds:
- **OneStopEnglish (Classification):** Accuracy (%), Macro F1-Score, and Quadratic Weighted Kappa (QWK).
- **CLEAR (Regression):** Root Mean Squared Error (RMSE) and Pearson correlation ($r$).

Every run automatically logs library versions (`torch`, `transformers`, `spacy`), random seeds, deterministic CUDA flags, confusion matrices / error residuals, and wall-clock runtime.

---

## 6. Changelog & Future Scaling
- **`v1.0`:** Initial protocol covering OneStopEnglish (GroupKFold) and 30 generic features.
- **`v1.1` (Current):** Added `StratifiedKFold` vs. `GroupKFold` leakage comparison, added CLEAR continuous benchmark, grouped features into 3 auditable families ($d = 22$), and added Experiment B (Feature Ablation).
- **Planned Extensions (Post-Core Results):**
  - *Cross-Dataset Generalization:* Train on CLEAR and test zero-shot/mapped transfer on OneStopEnglish.
  - *Fusion Mechanism Check:* Compare direct concatenation against a gated/MLP projection layer if feature scale imbalance occurs.