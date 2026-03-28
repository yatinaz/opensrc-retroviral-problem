# Interview Cheatsheet — Mandrake Retroviral Wall Challenge

> For Qure.ai interview. Know these numbers cold. Know the "why" behind each decision.

---

## KEY NUMBERS (memorise these)

| Metric | Value |
|--------|-------|
| Dataset size | **57 RTs** (21 active, 36 inactive) |
| RT families | **7** (LOFO = 7 folds) |
| CLS score (v14) | **0.8034** |
| PR-AUC | **0.811** |
| Weighted Spearman | **0.7959** |
| Beta (LR rank weight) | **0.66** |
| Strongest feature | **thumb_fident** (Spearman ρ = 0.901 with PE efficiency) |
| 2nd strongest feature | **n_hairpins_found** (ρ = 0.816) |
| ESM-2 PCA components | **5** (fitted inside each fold) |
| Submission deadline | April 30, 2026 (submitted early) |

---

## CLS METRIC — what it is and why it matters

**CLS = 2 × PR-AUC × WSpearman / (PR-AUC + WSpearman)**

- Harmonic mean — penalises imbalance. A model scoring 1.0 PR-AUC but 0.5 WSpearman gets CLS = 0.667.
- PR-AUC: rewards finding all 21 active RTs without false positives (relevant for small positive class).
- Weighted Spearman: rewards ranking the 21 actives in the right efficiency order, with high-efficiency RTs weighted more heavily.

---

## FEATURE CATEGORIES AND WHY EACH MATTERS

| Category | Key Features | Why It Matters |
|---|---|---|
| FoldSeek TM-scores | `foldseek_best_TM`, `foldseek_TM_MMLV`, `foldseek_TM_HIV1` | Structural similarity to known active RTs is the primary activity signal. TM-score > 0.5 = same fold. |
| Biophysical | `camsol_score`, `instability_index`, `D1_D2_dist`, `native_net_charge` | Protein stability, solubility, and active-site geometry determine whether the enzyme folds and functions. |
| Thermal stability | `t55_raw`, `t60_raw`, `t65_raw` | Thermostable proteins tend to be better expressed and more active. |
| Engineered flags | `is_broken_enzyme`, `is_no_thumb`, `is_no_hairpin` | Binary "missing domain" signals — an RT without a thumb subdomain cannot engage the template efficiently. |
| New structural | `thumb_fident` (ρ=0.901), `n_hairpins_found` (ρ=0.816), `isoelectric_point`, `n_salt_bridges`, `hbonds_per_res`, `n_strands_total` | The thumb subdomain identity to MMLV-RT is the strongest single predictor. These were absent from the v9 baseline and produced the biggest gains. |
| ESM-2 PCA | `pca_0..pca_4` | Language model embeddings encode evolutionary context. 5 PCA dims are enough; fitting inside each fold prevents leakage. |

---

## THE 3 HARDEST INTERVIEW QUESTIONS

### 1. "Why LOFO instead of standard k-fold CV?"

**Answer**: The 57 RTs span 7 evolutionary families. If you randomly split into k folds, you'll almost certainly have RT family members in both train and test. The model then learns a family-level shortcut: "this looks like a Retroviral RT → probably active." LOFO forces the model to generalise *across* evolutionary lineages — it must predict how a held-out family behaves based on structural and biophysical features learned from the other 6 families. That's the real test: given a new RT from a family we've never tested, can we predict whether it works?

This is directly analogous to **site-level leakage in medical imaging**: if you split patient data randomly across scanner sites, the model can overfit to scanner artifacts. You need site-level (family-level) splits to measure true generalisation.

---

### 2. "Why rank blending instead of probability averaging?"

**Answer**: LR produces probabilities in [0, 1] calibrated to the binary classification task. XGB/CB produces predicted PE efficiency in [0, 50+] calibrated to the regression task. Direct averaging is dimensionally incoherent — a 0.9 LR probability and a 0.9% efficiency prediction are not the same thing on any scale.

Converting both to fractional ranks [0,1] solves this:
- Ranks are always on the same [0,1] scale regardless of the original output's magnitude.
- Ranks are robust to outliers in the regression output (a single RT predicted at 200% efficiency doesn't dominate the blend).
- The beta parameter has a clear interpretation: beta=1.0 means "use only classification", beta=0.0 means "use only regression", and values in between blend the signals.

Practically: beta=0.66 was found by sweeping 0.00–1.00 in 0.01 steps under LOFO CV. The classifier is dominant because the binary signal (active/inactive) is strong (TM-score alone separates families), while the regressor adds the efficiency ordering among actives.

---

### 3. "What would you do next to improve further?"

**Specific next steps, in priority order:**

1. **More structural features**: Compute the angle between the thumb and palm subdomains (known to affect template processivity). Extract the active-site pocket volume and electrostatics. These require parsing the AlphaFold2 PDB files that are already available.

2. **Protein language model fine-tuning**: Fine-tune ESM-2 on a curated RT sequence + activity dataset (UniProt + this challenge). The current approach uses frozen embeddings. Even LoRA fine-tuning on 100–200 labelled RTs from the literature could improve the PCA components.

3. **Data augmentation via homologues**: For each of the 57 RTs, retrieve 5–10 close homologues (via MMseqs2, 90% identity cutoff) with known activity annotations from literature. This could double the effective training set size for the regressor.

4. **Ensembling across CV seeds**: The model's performance varies slightly with random seed due to small n. Averaging predictions from 5 seeds with different data orderings would reduce variance.

5. **Calibrated stacking**: Train a meta-learner (Ridge regression) on clf_oof and reg_oof OOF predictions rather than hand-tuning beta. This might discover non-linear combinations.

---

## BRIDGES TO QURE.AI

These map the Mandrake challenge to problems Qure.ai faces daily:

| Mandrake Challenge | Qure.ai Equivalent |
|---|---|
| 57 samples, tiny dataset | Rare diseases, rare imaging findings — small labelled datasets |
| Family-level LOFO splits | Site/scanner/hospital-level splits to prevent scanner-artifact leakage |
| Rank blending across miscalibrated models | Ensemble of models trained on different CT scanners or imaging modalities |
| `thumb_fident` as biology-informed feature | Anatomy-guided ROI features (e.g., lung lobe segmentation) vs raw image features |
| CLS metric — harmonic mean of precision and ranking | Balanced precision/recall tradeoff in clinical triage (false negatives = missed disease, false positives = unnecessary follow-up) |
| PCA inside each fold (no leakage) | Normalising image statistics per-scanner-site inside each hospital fold |

---

## ONE-LINE ELEVATOR PITCH

> "I built an ML pipeline to predict which reverse transcriptase enzymes work for prime editing, using protein structure embeddings and a rank-blended LR + XGB/CatBoost ensemble evaluated with rigorous Leave-One-Family-Out cross-validation — achieving CLS = 0.8034."

---

## COMMON FOLLOW-UP QUESTIONS

**"What is your CLS and how does it compare to baselines?"**
v9 (XGB/CB baseline): 0.677. v14 (this solution): 0.8034. Delta = +0.126. The biggest single jump was v9→v10 (+0.105) from switching the classifier to LR and discovering `thumb_fident`.

**"Why not use a neural network?"**
With n=57, a neural network would overfit catastrophically. LR with L2 regularisation is the right prior for linear separability at this scale. The ESM-2 embeddings provide the non-linear "neural network" component without training on this tiny dataset.

**"Why XGBoost + CatBoost and not just one?"**
They make different errors — XGBoost is more aggressive on splits, CatBoost is more conservative. Their average OOF predictions are better calibrated than either alone. The correlation between their OOF errors is ~0.7, meaning the ensemble picks up genuine signal from the disagreement.

**"What does `thumb_fident` measure?"**
Sequence identity (0–1) between the thumb subdomain of a query RT and the MMLV-RT thumb subdomain. MMLV-RT is the canonical high-efficiency prime editing RT. High thumb identity = similar template-gripping mechanism = similar PE efficiency. It is NaN for 17 RTs that have no detectable thumb domain — those are mostly non-Retroviral families and are all inactive.
