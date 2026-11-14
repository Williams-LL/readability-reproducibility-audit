# Reproducibility & Stress-Test Protocol
**Target Paper:** Lee et al. (2021). *Pushing on Text Readability Assessment: A Transformer Meets Handcrafted Linguistic Features* (EMNLP 2021).  
**Author:** [Tu Nombre]  
**Hardware Environment:** Local NVIDIA GeForce RTX 2060 (6GB VRAM) / Google Colab T4 (16GB VRAM)  

---

## 1. Objective & Scope
Independently re-implement the hybrid `RoBERTa + Linguistic Features` architecture in pure PyTorch (without relying on the authors' codebase) and audit whether the reported gains over a text-only Transformer remain statistically significant under:
1. Multi-seed variance (5 random seeds).
2. Data scarcity regimes (simulating low-resource educational domains).
3. Lightweight model substitution (simulating low-compute school environments).

---

## 2. Controlled Deviations from the Original Paper
To ensure full transparency and deterministic execution, the following methodological choices are fixed:
* **Data Splitting (Leakage Prevention):** Because OneStopEnglish consists of 189 parallel article triplets across 3 levels (567 texts), we use **5-Fold Group Cross-Validation** grouped by article ID rather than standard random/stratified splits, preventing topic leakage across folds.
* **Linguistic Feature Set ($d = 30$):** Instead of the 255 external features used in the original paper, we implement a self-contained, deterministic Python pipeline (`spaCy` `en_core_web_sm`) extracting 30 core features across four categories: surface counts, traditional readability formulas (FKGL, SMOG, CLI, ARI), lexical diversity (TTR, root-TTR), and syntactic/POS distributions.
* **Feature Scaling:** `StandardScaler` is fitted strictly on the training folds at each split and applied to validation/test folds to prevent distribution leakage.
* **Optimization & Memory:** Effective `batch_size = 16` (implemented as physical `batch_size = 8` with `gradient_accumulation_steps = 2` and `fp16` mixed precision to fit within 6GB VRAM). Optimizer: `AdamW` (`lr = 2e-5`, `weight_decay = 0.01`, linear warmup over 10% of steps, 5 epochs).

---

## 3. Experimental Matrix (Stress Tests)

### Experiment A: Baseline vs. Hybrid Replication (Full Dataset - 100%)
* **Models:**
  * `M1 (Baseline):` `roberta-base` + Linear Classification Head.
  * `M2 (Hybrid):` `roberta-base` + 30 Linguistic Features + LayerNorm + Linear Head.
* **Protocol:** 5 Folds $\times$ 5 Random Seeds (`42, 100, 2024, 777, 999`) = 25 runs per model.
* **Goal:** Verify if `M2 > M1` holds across seeds and compute paired statistical significance ($p < 0.05$).

### Experiment B: Low-Resource Data Scarcity Test
* **Condition:** Subsample the training folds to **25%** and **50%** of the original articles (keeping test folds intact at 100%).
* **Protocol:** Compare `M1` vs. `M2` across 5 Folds $\times$ 3 Seeds.
* **Hypothesis:** Handcrafted linguistic features provide stronger regularization and greater accuracy gains when training data is scarce.

### Experiment C: Compute-Constrained Educational Deployment
* **Condition:** Replace `roberta-base` (125M params) with `distilbert-base-uncased` (66M params).
* **Protocol:** Compare `DistilBERT` vs. `DistilBERT + Linguistic Features` across 5 Folds $\times$ 3 Seeds.
* **Goal:** Measure inference speed, VRAM footprint, and whether linguistic features recover accuracy lost by model compression.

---

## 4. Evaluation Metrics & Logging Standards
For every experiment, we will report **Mean $\pm$ Standard Deviation** across folds and seeds for:
1. **Accuracy (%)** (Direct comparison with Lee et al., 2021).
2. **Macro F1-Score** (Class-balanced performance).
3. **Quadratic Weighted Kappa (QWK)** (Penalizes distant ordinal errors, e.g., misclassifying *Elementary* as *Advanced*).

All runs will automatically log PyTorch/Transformers versions, CUDA deterministic flags, confusion matrices, and exact training wall-clock time.