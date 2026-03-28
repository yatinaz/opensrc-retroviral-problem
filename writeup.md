# Retroviral Wall Challenge — Solution Writeup (v14)

**LOFO CLS = 0.8034** | PR-AUC = 0.811 | Weighted Spearman = 0.7959

---

## Problem

Binary classification of 57 RT enzymes (21 active, 36 inactive) + ranking active RTs by prime editing efficiency. Evaluated with CLS = harmonic mean of PR-AUC and Weighted Spearman. Cross-validation uses Leave-One-Family-Out (LOFO) over 7 RT families.

---

## Approach

A two-head ensemble: a **Logistic Regression classifier** produces activity probabilities; an **XGBoost + CatBoost regressor ensemble** predicts PE efficiency. Final scores are a weighted rank blend:

```
final = 0.66 × rank(LR_prob) + 0.34 × rank((XGB + CB) / 2)
```

The optimal beta (0.66) was found by sweeping 0.00–1.00 in 0.01 steps under LOFO CV.

---

## Features

### Classifier features
- FoldSeek structural similarity TMs (vs HIV-1, MMLV, MMLVPE, Telomerase, LTR Retrotransposon, Group 2 Intron, Retron)
- Biophysical: `triad_best_rmsd`, `D1_D2_dist`, `pocket_hydrophobic_per_res`, `camsol_score`, `perplexity`, `instability_index`, `native_net_charge`, `protein_length_aa`, `TM_ratio_MMLVPE_telo`
- Structural: `hbonds_per_res` (r = −0.553 with active), `n_strands_total` (r = +0.351 with active)
- Engineered flags: `is_broken_enzyme`, `length_to_TM_ratio`, `is_no_thumb`, `is_no_hairpin`
- ESM2 PCA (5 components, **fitted inside each LOFO fold** to prevent leakage)

### Regressor features
All classifier features plus:
- `thumb_fident` (Spearman ρ = +0.901 with PE efficiency)
- `n_hairpins_found` (ρ = +0.816 with PE efficiency)
- Thermal stability: `t55_raw`, `t60_raw`, `t65_raw`
- `isoelectric_point` (r = −0.523 with PE efficiency among actives)
- `n_salt_bridges` (r = +0.385 with PE efficiency among actives)

---

## Key Insights

1. **LR beats tree classifiers** at this scale. With n=57 and near-linear separability (`foldseek_best_TM` alone gives AUC ≈ 0.78), LR uses all features simultaneously vs. depth-2 trees that select only 2 features per branch.

2. **`thumb_fident` is the strongest regressor signal** — the thumb subdomain identity to a known RT is nearly perfectly correlated with PE efficiency (ρ = 0.901). Absent from baseline features.

3. **PCA inside each fold** is critical. Fitting PCA on all 57 embeddings before splitting inflates LOFO performance due to test-set leakage in the embedding space.

4. **NaN encoding matters**: tree models use −999 fill (allowing the model to learn "missingness" as a signal); LR uses median imputation + z-scaling.

---

## Score Progression

| Version | CLS    | Change  | Key addition |
|---------|--------|---------|--------------|
| v9      | 0.677  | —       | XGB/CB baseline |
| v10     | 0.7824 | +0.105  | LR clf, thumb_fident, n_hairpins_found, PCA fix |
| v11     | 0.7908 | +0.008  | hbonds_per_res in CLF |
| v14     | 0.8034 | +0.013  | n_strands_total in CLF, isoelectric_point + n_salt_bridges in REG, beta=0.66 |

---

## What Was Tried But Didn't Help

- LightGBM and Ridge regressors — hurt ensemble WSpearman
- Additional linear regressors (despite good individual correlations, averaging with XGB/CB degraded scores)
- Adding `hbonds_per_res` and `salt_per_res` to tree classifiers — hurt performance
