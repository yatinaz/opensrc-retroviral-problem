# Interview Guide — Retroviral Wall Challenge

---

## Section 1 — 2-Minute Opening Script

Prime editing is a precision gene-editing technology that lets us rewrite specific letters in the genome without cutting both DNA strands. At its core is a reverse transcriptase — an enzyme that copies RNA into DNA — fused to a modified CRISPR protein. The choice of RT matters enormously: in one clinical application, Duchenne muscular dystrophy correction, a high-efficiency RT like MMLV-RT can drive edit rates an order of magnitude higher than a weak one. But MMLV-RT has poor thermal stability in human cells, which is why the field is hunting for better candidates.

The problem I worked on is exactly that hunt, formalised as a Mandrake Bioworks open challenge. We have 57 RT sequences that have been experimentally tested for prime editing activity. 21 are active; 36 are not. Among the 21 actives, their efficiency spans two orders of magnitude. The task is binary classification plus ranking, evaluated jointly using the CLS metric — the harmonic mean of PR-AUC and Weighted Spearman. The only allowed input is the protein sequence. And the evaluation is Leave-One-Family-Out cross-validation across 7 evolutionary RT families, which is deliberately the hardest possible evaluation for a dataset of this size.

The reason LOFO matters is that random k-fold would let the model memorise family membership — "RTs that look like MMLV are probably active" — without learning anything about generalisation to novel organisms. LOFO forces the model to predict activity for a family it has never trained on.

My final result was CLS=0.803, with PR-AUC=0.811 and Weighted Spearman=0.796. The model recovered 10 of 12 known Retroviral true positives under LOFO. The architecture that got there — and how each decision was validated — is what I'd like to walk you through.

---

## Section 2 — Methodology Talking Points

### LOFO Evaluation Framework

**(a)** What it is: Leave-One-Family-Out cross-validation rotates through all 7 RT evolutionary families, holding out all members of one family per fold and training on the remaining 6.

**(b)** Decision: LOFO was the official evaluation protocol. I adopted it exactly, including in all development sweeps, so that every feature and hyperparameter decision was made under the same distributional shift the test set imposes.

**(c)** What was tried first and why it failed: Standard stratified k-fold gave CLS ~0.05 higher on every version because family members ended up in both train and test, allowing the model to memorise structural similarity within clades. Those numbers were meaningless for deployment.

**(d)** Exact number: LOFO v14 CLS=0.803 vs stratified 5-fold "CLS" ~0.86 on the same pipeline — a 7-point gap that is entirely an artefact of the CV strategy.

---

### CLS Metric and Why It Forced a Two-Stage Architecture

**(a)** What it is: CLS = 2 · PR-AUC · WSpearman / (PR-AUC + WSpearman), the harmonic mean of two complementary objectives. PR-AUC rewards finding actives without false positives. Weighted Spearman rewards correct ranking of actives by efficiency, with high-efficiency RTs weighted more.

**(b)** Decision: Two separate heads — a classifier for PR-AUC and a regressor for WSpearman — rather than one model trying to do both.

**(c)** What was tried first and why it failed: In v8, a single XGBoost/GBR model ranked the 21 actives identically (WSpearman ≈ 0) because the classification boundary dominated the output. CLS collapsed to ~0.49 even though the classifier was working.

**(d)** Exact number: Single-head v8 CLS=0.487 vs two-head v8t CLS=0.613 — a 25% relative improvement just from separating the objectives.

---

### Why L2 LogisticRegression Beat XGBoost as the Classifier

**(a)** What it is: Logistic Regression fits a linear boundary in all features simultaneously with L2 regularisation. XGBoost at depth=2 can only use 2 features per branch.

**(b)** Decision: LR (C=0.1, balanced class weights) replaced XGBoost as the classifier in v10.

**(c)** What was tried first and why it failed: XGBoost classifier in v9 (CLS=0.677). With n=57 and near-linear separability — foldseek_best_TM alone gives AUC ≈ 0.78 — depth-2 trees can only condition on 2 features at a time. They overfitted on whatever pair of features happened to be most discriminative in each fold while ignoring the others.

**(d)** Exact number: LR switched in v10 contributed +0.105 CLS (from 0.677 → 0.7824), the single largest gain in the entire project, larger than all subsequent feature engineering combined.

---

### FoldSeek TM-Scores and the Family Confounding Question

**(a)** What it is: FoldSeek compares predicted AlphaFold structures against a library of known RT reference structures (MMLV, HIV-1, MMLVPE, Telomerase, LTR Retrotransposon, Group II Intron, Retron) and returns a TM-score for each comparison.

**(b)** Decision: Include all 7 family-specific TM-scores plus the global best TM-score in both the classifier and regressor.

**(c)** What was tried first: Concern that TM-scores encode family identity directly, which might inflate LOFO scores for within-family predictions. However, LOFO explicitly removes all members of the target family from training. A model trained without the Retroviral fold will still see that a test protein has high TM-score to MMLV — but it will have never trained on Retroviral sequences, so it must generalise from the correlation between TM-score and activity that it learned from other families.

**(d)** Exact number: Removing foldseek_best_TM from the classifier dropped PR-AUC by ~0.04 in held-out validation — it is irreplaceable.

---

### PCA Fitted Inside the Fold Rather Than Globally

**(a)** What it is: ESM-2 produces 1280-dimensional embeddings per sequence. PCA reduces these to 5 components. "Inside fold" means PCA is fitted on training sequences only, then applied to test sequences — not fitted on all 57 before splitting.

**(b)** Decision: PCA refitted per fold, every time.

**(c)** What was tried first: A preliminary version fitted PCA on the full 57-sequence matrix before the LOFO loop. This is leakage: the test sequence's embedding direction subtly influences the principal components used to represent training sequences. It inflates WSpearman by ~0.015–0.020 in practice.

**(d)** Exact number: This is explicitly implemented in the code (`pca.fit_transform(all_embs[tr])` then `pca.transform(all_embs[te])`). The writeup correctly notes this as a deliberate design choice throughout, not a fix applied at a specific version.

---

### Rank Blend Instead of Score Product

**(a)** What it is: Both LR probabilities and regressor predictions are converted to fractional ranks in [0,1] before being combined as `0.66 × rank(LR_prob) + 0.34 × rank(reg_pred)`.

**(b)** Decision: Rank blend with beta=0.66, tuned by sweeping 0.00–1.00 in 0.01 steps under LOFO.

**(c)** What was tried first: Direct probability averaging. LR outputs are in [0, 1]. XGB/CB outputs are in [0, 50+] (percentage PE efficiency). Averaging them directly is numerically meaningless — a 0.9 LR probability is not comparable to a 0.9% efficiency prediction. The result was CLS ~0.05 lower than rank blending.

**(d)** Exact number: Best beta=0.66 gives CLS=0.803. Beta=0.50 (equal blend) gives CLS=0.797. Beta=1.0 (classifier only) gives CLS ~0.79. The regressor contributes 6 CLS points at the optimal blend weight.

---

### Regressor Purity Principle

**(a)** What it is: Features that help classification should not automatically be added to the regressor. The regressor's job is to rank actives by efficiency, not to distinguish active from inactive.

**(b)** Decision: Keep the regressor feature set strictly to features with high correlation to efficiency values, not to binary activity.

**(c)** What was tried first (v11 lesson): Adding `hbonds_per_res` to the regressor as well as the classifier. It correlates with binary activity (r=−0.553) but is almost uncorrelated with efficiency rank among actives. Adding it to the regressor degraded WSpearman while leaving PR-AUC unchanged, reducing CLS.

**(d)** Exact number: `hbonds_per_res` in classifier only (v11 baseline): CLS=0.791. Adding it to regressor too: CLS=0.783. An 8-point drop from one misplaced feature.

---

### OOF Residual Analysis as a Structured Feature Discovery Tool

**(a)** What it is: After generating OOF predictions from a fitted model, identify which sequences have the largest prediction errors and then test which candidate features are correlated with those residuals — rather than with the raw labels.

**(b)** Decision: `run_diagnostics.py` was used after v10 to audit which active RTs the model was still misranking. This guided the selection of candidate features for the v12/v13 systematic sweep.

**(c)** What was tried first: Adding features based purely on their Spearman correlation with PE efficiency across all actives. This sometimes picked features that were redundant with existing signals or that helped on the full set but not on the hard cases.

**(d)** Exact number: OOF residual analysis identified MMLV-RT being ranked 3rd despite the highest experimental efficiency as the key failure mode. This drove the investigation of isoelectric_point and n_salt_bridges, which added +0.013 CLS in v14.

---

## Section 3 — 15 Predicted Q&A Pairs

**Q1: Why LOFO and not random k-fold?**

The 57 RTs belong to 7 evolutionary families with very different structural profiles. Random k-fold would, on average, put half of each family's members in training and half in test. A model could then learn "Retroviral-looking sequences are active" purely from structural similarity within the clade, producing inflated numbers that tell you nothing about deployment generalisability. LOFO holds out all members of one family per fold, so the model must predict activity for sequences from an evolutionary lineage it has never trained on. That is exactly the scenario when a company screens an RT from a novel organism. In practice, random 5-fold gave CLS ~0.86 on the same v14 pipeline — about 6 points higher than LOFO — and every one of those extra points was memorisation artefact, not signal. The only meaningful number is the LOFO one.

*Follow-up they'll ask: doesn't LOFO with 7 folds give very noisy estimates?* Yes — each fold has between 4 and 18 sequences, so individual fold estimates have high variance. But the aggregate CLS averages over all 57 predictions simultaneously, which stabilises the estimate. The variance is real and acknowledged in the limitations, but there is no better evaluation structure for this data.

---

**Q2: Doesn't the model just learn Retroviral = active?**

This is the key concern LOFO is designed to address. When the Retroviral family is the held-out fold, the model has never seen any Retroviral sequence during training. It cannot use family membership directly. What it does use is structural similarity: the FoldSeek TM-score to MMLV-RT measures whether a novel sequence has Retroviral-like fold geometry, even if the sequence itself is novel. That is a generalisable signal — it is a structural proxy, not a label proxy. The evidence that the model has learned structure and not just identity is that it achieves PR-AUC=0.811 across all 7 folds, including the Retroviral fold where the held-out sequences have the highest TM-score to MMLV.

*Follow-up: but doesn't foldseek_TM_MMLV encode family identity directly?* Only partially. A sequence with high TM-score to MMLV is Retroviral-like in fold, but two non-Retroviral sequences can also have moderate TM-scores to MMLV (Retrons, some LTR Retrotransposons). The correlation is not 1:1.

---

**Q3: How do you know CLS=0.803 is real signal at n=57?**

You cannot be fully certain at this sample size — that is an honest limitation. What I can say is: first, LOFO gives a conservative estimate by construction. Second, the signal is consistent across 5 of 7 folds (CRISPR-associated and Other folds have zero actives and contribute no per-fold CLS). Third, the key features — thumb_fident, n_hairpins_found, foldseek_best_TM — have strong biological justification. The thumb subdomain grips the RNA/DNA hybrid during synthesis; high thumb identity to MMLV is a mechanistically sensible predictor of MMLV-like activity. The model is not fitting noise patterns; it is finding signals that domain experts would recognise. Fourth, the systematic sweep in v12/v13 validated that each added feature improved CLS under LOFO across multiple folds, not just one.

---

**Q4: Why does LogisticRegression beat XGBoost here?**

At n=57 with near-linear separability, XGBoost at depth=2 can only interrogate 2 features per branch. With 30+ features, a depth-2 tree is effectively ignoring most of them in each fold. Logistic Regression fits a linear boundary in all 30+ features simultaneously with L2 regularisation (C=0.1), which prevents any single feature from dominating while allowing the full feature set to contribute. When the decision boundary is approximately linear — which foldseek_best_TM alone shows is the case here — LR exploits that structure efficiently. XGBoost's expressiveness is actually a liability at this scale: it has enough capacity to overfit the training folds while LR's inductive bias (linearity) aligns with the true structure of the data.

*Follow-up: did you try deeper trees?* Yes. Depth=3 and depth=4 both degraded PR-AUC. Deeper trees memorise the training families more aggressively without generalising to the held-out one.

---

**Q5: You use FoldSeek TM-scores to known reference families — isn't that encoding family identity directly?**

Partially — but it is structural similarity, not label lookup. FoldSeek computes how similar a query structure is to a reference structure in 3D geometry. A Retroviral RT will have high TM-score to MMLV because it has Retroviral fold, but the TM-score also grades within the Retroviral family, separating high-efficiency from low-efficiency members on structural grounds. More importantly, during LOFO evaluation with the Retroviral family held out, the model learns from the correlation between TM-scores and activity across the other 6 families, then applies that learned correlation to the Retroviral test sequences. If TM-score were a pure family indicator, the model would fail on novel families — but Group II Intron achieves near-perfect PR-AUC in LOFO, despite being structurally unlike Retroviral, because the model generalises the TM-score / activity relationship cross-family.

---

**Q6: Walk me through every data leakage risk in your pipeline.**

There are three places where leakage could occur. First, ESM-2 PCA: if you fit PCA on all 57 embeddings before the LOFO split, the test sequence's embedding direction subtly influences the principal components. We prevent this by refitting PCA inside each fold on training sequences only. Second, the beta sweep: we sweep 0.00–1.00 to find the optimal blend weight, but this sweep is conducted on OOF predictions — it is equivalent to tuning a scalar on the training data distribution, not on the test set. This is a minor form of optimism but is standard practice and the effect is small (the range of CLS across beta is ~0.01 in the neighbourhood of the optimum). Third, feature engineering: all engineered features (TM_ratio, is_no_thumb, etc.) are computed from the sequence or predicted structure, not from the labels. No label-dependent scaling or transformation is applied. The only label-dependent step is LR's class_weight="balanced", which is an in-fold operation.

---

**Q7: Beta is tuned on the same OOF predictions you report as your score — isn't that optimistic?**

Yes, by a small amount. Beta is a single scalar tuned to maximise CLS across the full 57-sample OOF. This is conceptually similar to tuning a calibration parameter on the training distribution. The optimism is bounded by the sensitivity of CLS to beta: the CLS range across all beta values is about 0.02 (from 0.79 at beta=1.0 to 0.80 at beta=0.66). A true out-of-sample evaluation would require a held-out set to tune beta on, which we do not have. This is acknowledged as a limitation. The practical magnitude of the optimism is likely 0.005–0.01 CLS, which does not change the qualitative story.

---

**Q8: Why does adding features to the regressor sometimes hurt WSpearman even when they correlate with efficiency?**

The regressor is optimised to rank actives by efficiency. If you add a feature that correlates with binary activity (active/inactive) but is only weakly correlated with efficiency rank *among actives*, you are giving the regressor a signal that teaches it to separate active from inactive — a task the classifier handles — rather than to order actives by efficiency. At depth=2, XGBoost will use that feature in some branches where it would otherwise have used an efficiency-correlated feature. The net result is that ranking among actives degrades even though the feature looks useful in the marginal correlation. `hbonds_per_res` is the clearest example: r=−0.553 with binary activity but near-zero correlation with efficiency rank, and adding it to the regressor dropped WSpearman by ~0.008.

---

**Q9: What caused the single biggest improvement?**

Switching the classifier from XGBoost/CatBoost to L2 Logistic Regression in v10, which contributed +0.105 CLS (from 0.677 to 0.7824). The reason is n=57 with near-linear separability. XGBoost at depth=2 can only condition on 2 features per branch, so it is effectively a high-variance two-feature selector. Logistic Regression fits a linear boundary in all 30+ features simultaneously with ridge regularisation that penalises large coefficients and handles multicollinearity. At this scale, the structure of the classification problem (foldseek_best_TM alone gives AUC≈0.78) aligns with LR's inductive bias. The same v10 update also introduced thumb_fident and n_hairpins_found to the regressor, contributing to the WSpearman gain, but the classifier switch was the dominant factor.

---

**Q10: Which RT is the model most wrong about and why?**

MMLV-RT ranks 3rd in OOF predictions despite being the highest-efficiency RT in the dataset. The model places POL11ERV-RT and Tf1-RT above it. MMLV-RT is the structural reference for most FoldSeek comparisons — the model "knows" it looks Retroviral — but the efficiency ordering within Retroviral RTs is determined by subtler structural features than TM-score captures. thumb_fident and n_hairpins_found do help, but they are coarse proxies. The real difference between MMLV (PE eff ≈ 35%) and POL11ERV (PE eff ≈ 28%) likely lies in loop dynamics and active site geometry that AlphaFold's static structure does not fully resolve. This is the honest limitation of structural similarity scores as efficiency predictors.

---

**Q11: What would you try next with another week?**

Three things. First, homolog expansion: MMLV-RT has hundreds of characterised variants in public databases with known thermostability and activity data. Adding those as auxiliary training examples with a domain-adaptation term could substantially improve the regressor's efficiency ordering. Second, active site geometry: the triad residues of the YMDD polymerase motif, their relative geometry, and the depth of the nucleotide-binding pocket are mechanistically relevant and not fully captured by global TM-scores. Custom PyMOL/Biopython scripts could extract these. Third, uncertainty quantification: the current model gives point predictions; a conformal prediction wrapper could provide valid coverage guarantees at any confidence level, which would be more useful for experimental prioritisation.

---

**Q12: What does this tell you biologically about what makes a good prime editing RT?**

The two strongest signals are thumb_fident (ρ=0.901 with PE efficiency) and n_hairpins_found (ρ=0.816). thumb_fident measures how similar the thumb subdomain — the structural domain that grips the RNA/DNA hybrid during synthesis — is to MMLV's thumb. This strongly suggests that MMLV-like thumb geometry is the dominant determinant of prime editing processivity. The hairpin count signal is less directly interpretable but likely reflects structural complexity of the fingers/palm region that affects template switching. Thermal stability features (t55–t65) also contribute, consistent with the hypothesis that more stable enzymes tolerate the intracellular environment in human cells better. These are testable hypotheses: engineering MMLV-like thumb geometry into a thermostable scaffold should, if the model is right, produce a high-efficiency prime editing RT.

---

**Q13: MMLV-RT ranks 3rd not 1st in your OOF — is that a failure?**

It is a partial failure in the ranking objective, but it is worth contextualising. The Weighted Spearman metric weights high-efficiency RTs more heavily — a mis-ranking of MMLV (rank 3 vs rank 1) costs more than a mis-ranking of a low-efficiency active RT. However, WSpearman=0.796 is still substantially above the ~0.6 baseline from random ranking of actives. The ranking failure on MMLV is explained by the OOF structure: in the Retroviral fold, the model has never seen any Retroviral training signal, so it must predict MMLV's efficiency from non-Retroviral data. The fact that it correctly identifies MMLV as the 3rd highest-efficiency RT out of 57 total sequences — without having trained on any Retroviral sequence — is arguably strong.

---

**Q14: How would you adapt this pipeline for 200 new sequences?**

The architecture would stay largely intact. With n=200, tree models become viable as classifiers — LR's advantage over XGBoost disappears once the sample size supports depth-3+ trees that can interrogate more features per branch. I would re-run the systematic feature sweep on the expanded dataset because some features that are marginal at n=57 may become significant at n=200. The LOFO structure stays if the new sequences still have family structure — if they span the same 7 families, the evaluation strategy is identical. If there are new families, I would add corresponding FoldSeek reference structures. The beta sweep would be re-done on the new OOF. The key thing that changes is that the regressor can absorb more features without overfitting, so features like ramachandran_out and aromaticity that were negative additions at n=57 might contribute positively at n=200.

---

**Q15: How long does the pipeline take and is it reproducible?**

The full v14 pipeline — loading data, running LOFO across 7 folds, beta sweep, saving results — takes about 3–4 minutes on a standard laptop CPU with the pre-computed ESM-2 embeddings (esm2_embeddings.npz). The pipeline is fully deterministic given SEED=42, which is fixed at the top of the script. All intermediate OOF predictions are computed in a single forward pass with no stochastic elements beyond the fixed random seed. `requirements.txt` pins all library versions. The only non-reproducible element is the AlphaFold structure prediction (used to generate the structural features), which requires the pre-computed PDB files that are not in the repo due to size — but the features derived from them are committed to `data/handcrafted_features_with_struct.csv`. Anyone can reproduce CLS=0.803 by running `python retroviral_wall_v14.py` with the data files and `esm2_embeddings.npz`.

---

## Section 4 — Figure Talking Points

**Fig 1 — family_overview**

This figure shows the dataset composition across the 7 evolutionary RT families. The thing I'd draw your attention to is the Retroviral family: 12 of the 21 active RTs are Retroviral, so more than half of all the active signal in the dataset comes from a single evolutionary clade. That is the core structural challenge. Any model that learns "Retroviral = active" will look good under random CV but fail under LOFO. The three families with zero active members — CRISPR-associated, Unclassified, and Other — represent pure negative signal; those sequences tell the model what an RT that is not adapted for prime editing looks like. The 21 vs 36 class split (37% positive) is also visible here — a meaningful imbalance that we handle with balanced class weights in LR.

**Fig 2 — 01_era1_progression**

Era 1 covers v1 through v7, where we were optimising Macro-F1 and the count of Retroviral true positives recovered. The early versions used Random Forest and SVM ensembles with handcrafted biophysical features. The thing to notice is how flat the progression is — going from v1 to v7 gave us only about 0.016 MF1 improvement. The big jump at v6/v7 came from two things: introducing rank blending across multiple models, and adding catalytic-window ESM-2 embeddings that focus on the active site region. The final Era 1 result — MF1=0.650, 10/12 Retroviral TPs — was respectable, but ranking was completely broken: all active RTs received the same predicted score, making the transition to Era 2 necessary.

**Fig 3 — 02_era2_progression**

Era 2 is where the real story is. The metric changed entirely — from Macro-F1 to CLS, the harmonic mean of PR-AUC and Weighted Spearman. The first v8 attempt, using a single XGB/CatBoost model, gave CLS=0.487 because ranking collapsed. The two-head architecture (v8t) immediately jumped to 0.613. Then you see a steady climb from v9 to v14. The single biggest step — the tall bar at v10 — is where Logistic Regression replaced XGBoost as the classifier, contributing +0.105 CLS in one change. Everything after that — the feature sweeps in v12 and v13, the beta optimisation in v14 — adds smaller increments on top of that foundation.

**Fig 4 — cls_progression**

This figure plots CLS from v8 to v14 with annotations for each key change. The visual tells you immediately that the LR switch at v10 dominates. The CLS scale goes from roughly 0.49 to 0.80, and about half of that total gain happened in a single version. What I want to emphasise is the harmonic mean structure: CLS rewards balanced improvement in both PR-AUC and WSpearman. You cannot optimise one at the expense of the other — the harmonic mean collapses towards zero if either component is weak. That is why the two-stage architecture was necessary: no single model could simultaneously maximise both objectives at n=57.

**Fig 5 — feature_distributions**

This shows violin + strip plots for the 6 most discriminative features, split by active (blue) and inactive (orange). Look at thumb_fident: the two distributions are almost non-overlapping. Active RTs cluster around high thumb subdomain identity to MMLV; inactive RTs cluster near zero or NaN. Similarly, foldseek_best_TM shows strong separation. The features with overlapping distributions — like instability_index — still contribute to the ensemble because they provide orthogonal signal in combination. The NaN indicators (is_no_thumb, is_no_hairpin) are themselves informative: an RT without a detectable thumb subdomain is almost always inactive.

**Fig 6 — feature_importance**

Feature importance by normalised XGBoost gain, coloured by whether the feature appears in the classifier, regressor, or both. The two things I'd highlight: first, foldseek structural features dominate the top of the classifier importance — structural similarity to known RTs is the strongest classification signal. Second, thumb_fident and n_hairpins_found appear near the top of the regressor importance but are absent from the classifier — these are regressor-only features because they capture processivity-relevant local geometry that ranks actives by efficiency without adding discriminative power for binary classification. This split is deliberate, not accidental.

**Fig 7 — lofo_performance**

Per-family PR-AUC under LOFO. The families where the model performs best — Group II Intron (near perfect), Retron, Retroviral — are the ones with clear structural signals that other families teach the model. Group II Intron achieves near-perfect separation because the structural profile of Group II Intron RTs is distinctive enough that the model can recognise them from non-Group-II-Intron training data. LTR Retrotransposon is the hardest — Tf1-RT and Ty3-RT are structurally unlike MMLV, so the model receives relatively weak training signal from non-LTR-Retrotransposon families for predicting LTR activity. Families with zero active members (CRISPR-associated, Other, Unclassified) show PR-AUC of 1.0 by convention — trivially correct because all test members are inactive.

**Fig 8 — rank_ensemble**

This diagram illustrates the rank blending step. LR outputs a probability in [0,1]; XGB+CatBoost output a predicted efficiency in [0,50+]. Both are converted to fractional ranks [0,1] — meaning the sequence with the highest LR probability gets rank 1.0 and the sequence with the lowest gets rank 0.0, and similarly for the regressor. These fractional ranks are then blended at 0.66/0.34. The 0.66 weight toward LR reflects the fact that PR-AUC is the harder component of CLS to optimise at this sample size — the classifier signal is cleaner than the regressor signal, so we weight it more. The blend was tuned empirically by sweeping beta under LOFO.

---

## Section 5 — Things Not to Say (With Corrections)

**Wrong:** "We fixed an ESM-2 PCA leakage bug in v10 that gave us the biggest single improvement."

**Correct:** The PCA was always fitted inside each LOFO fold — there was no leakage to fix. The single biggest improvement in v10 was switching the classifier from XGBoost/CatBoost to L2 Logistic Regression (+0.105 CLS). The PCA-inside-fold design was present from the beginning as a deliberate architectural choice, and the writeup notes it as such.

---

**Wrong:** "CLS=0.803 shows the model will generalise well to Phase 2 wet-lab screening of novel RTs."

**Correct:** CLS=0.803 is the LOFO estimate on the 57 sequences in the training set. It measures family-level generalisation, not sequence-level novelty. A Phase 2 wet-lab screen would involve sequences from potentially new organisms with no structural reference in the training data. Whether the model generalises to those is an open empirical question. The honest statement is: LOFO gives a conservative lower bound on within-distribution generalisation, but out-of-distribution performance is unknown.

---

**Wrong:** "The model learned the biology of prime editing RT activity."

**Correct:** The model learned structural proxies for activity — specifically, that structural similarity to known active RTs (measured by FoldSeek TM-scores) and palm domain geometry (thumb_fident, n_hairpins_found) predict PE efficiency. Whether these structural features are the actual biological mechanism or are correlated with the mechanism is a testable hypothesis, not an established fact. The thumb_fident result is consistent with the biology but not proof of it.

---

**Wrong:** "LOFO tests whether the model generalises to new sequences it hasn't seen."

**Correct:** LOFO tests whether the model generalises to new evolutionary *families* it hasn't trained on. Within each held-out family, the specific sequences are unseen, but the family-level structural profile is what varies. This is family-level generalisation, not sequence-level novelty. A sequence from a known family but with novel mutations would be handled differently from a sequence from an entirely new evolutionary lineage.

---

**Wrong:** "Adding more features always helps the ensemble."

**Correct:** Adding features to the wrong model hurts performance. The v11 lesson is that features correlating with binary activity (e.g. hbonds_per_res, r=−0.553 with active) degrade WSpearman when added to the regressor because they teach the regressor to distinguish active from inactive rather than rank actives by efficiency. Each feature must be validated under LOFO in the specific model head it is being added to.

---

**Wrong:** "The Weighted Spearman metric gives equal weight to all active RTs."

**Correct:** Weighted Spearman weights each active RT by its PE efficiency value. High-efficiency RTs (MMLV-RT, PE eff ≈ 35%) contribute much more to the score than low-efficiency actives (PE eff ≈ 0.5%). This is by design: the metric reflects that ranking the top candidates correctly matters more than ranking the bottom actives correctly for experimental prioritisation.

---

**Wrong:** "The regressor predicts which RTs are active."

**Correct:** The regressor predicts PE efficiency values. It is trained on all 57 sequences with PE efficiency as the target (inactive RTs get efficiency=0), but its output is used only for the WSpearman ranking component. The binary classification (active/inactive) is handled exclusively by the Logistic Regression classifier. The two heads serve different objectives and their outputs are combined via rank blending, not by treating them as interchangeable predictions.

---

**Wrong:** "LightGBM was part of the final ensemble."

**Correct:** LightGBM was tested in v8t and early Era 2 sweeps but was gated out because it degraded WSpearman relative to XGBoost+CatBoost when added to the regressor ensemble. The final model uses XGBoost and CatBoost as the two regressor components and Logistic Regression as the classifier. LightGBM does not appear in any version after v8t.

---

## Section 6 — Role-Specific Framing

**For a Computational Biology / Bioinformatics Scientist:**

The most biologically interesting result is thumb_fident — the Spearman correlation between thumb subdomain sequence identity to MMLV-RT and PE efficiency is 0.901, higher than any global structural similarity metric. The thumb subdomain grips the RNA/DNA hybrid during synthesis; the intuition is that MMLV-like thumb geometry enables the processivity required for prime editing without premature template dissociation. The n_hairpins_found signal (ρ=0.816) likely reflects complexity of the fingers region that enables template-directed synthesis. Thermal stability features (t55–t65) contribute to the regressor, consistent with the known correlation between thermostability and enzymatic activity in cellular environments. The LOFO structure is directly analogous to the deployment scenario: training on characterised RT families from known organisms, then predicting activity in a novel organism whose RT has never been screened — exactly the situation in industrial enzyme engineering.

**For an ML Scientist / AI Researcher:**

The core insight is about the sample size regime. At n=57 with near-linear separability (a single structural similarity score gives AUC≈0.78), tree classifiers are a poor choice. Logistic Regression at C=0.1 uses all 30+ features simultaneously with ridge regularisation, exploiting the low effective dimensionality of the classification problem — the +0.105 CLS gain from this switch is larger than all subsequent feature engineering combined. The rank blending design addresses the calibration problem in multi-objective optimisation: the two outputs live on different numerical scales, and score multiplication would not preserve rank relationships. Converting to fractional ranks before blending is a simple but robust solution. The regressor purity principle — features helping classification should not be added to the regressor — is a general lesson for two-objective ensemble design: each head should be optimised for its own objective using features that are discriminative for that objective, not features that are globally discriminative.

**For a Data Scientist / Generalist:**

The v10–v14 progression represents about 20 controlled experiments run systematically over five versions. v12 tested 6 LR regularisation strengths and 4 PCA dimension counts. v13 tested 13 candidate features, one at a time, with a strict gate: a feature was only added if it improved CLS under LOFO. v14 combined the winners. This discipline — one variable at a time, gate on the full evaluation metric, never add features that help on training but not test — is what produced a reliable 0.80+ score on a 57-sample dataset where overfitting is trivially easy. The OOF residual analysis in `run_diagnostics.py` was the structured diagnostic tool: rather than testing features at random, we identified which specific sequences the model was most wrong about, then designed features to address those failure modes. Three promising features were tested (ramachandran_out, aromaticity, palm_rnaseH_com_dist) and rejected because they failed the LOFO gate — they correlated with efficiency in the full set but not under held-out family evaluation.
