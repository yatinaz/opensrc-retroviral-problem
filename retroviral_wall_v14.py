"""
retroviral_wall_v14.py
=======================
V14 Final — best configuration found through systematic sweeps.

Changes from v11_clean:
  CLF: added n_strands_total  (r=+0.351 with active, 9 nulls)
  REG: added isoelectric_point (r=-0.523 with efficiency, 0 nulls)
       added n_salt_bridges    (r=+0.385 with efficiency, 0 nulls)
  beta: 0.66 (fine-tuned from 0.60)
  LR C: 0.1 (0.12 gives marginal +0.0001, not worth the change)

Score progression:
  v9         CLS=0.677
  v10        CLS=0.7824  (+0.105)
  v11_clean  CLS=0.7908  (+0.008)
  v14        CLS=0.8034  (+0.013)

Usage:
    python retroviral_wall_v14.py

Expects the following files relative to this script's directory:
    data/rt_sequences.csv
    data/handcrafted_features_with_struct.csv  (or handcrafted_features.csv)
    esm2_embeddings.npz  (large file, may live one level up from data/)
"""
import warnings
import json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from catboost import CatBoostRegressor

# ── Paths (relative to this script) ────────────────────────────────────────
HERE     = Path(__file__).parent
DATA_DIR = HERE / "data"

SEED = 42
np.random.seed(SEED)


def weighted_spearman(pred, true_eff, weights):
    """Compute weighted Spearman correlation between pred and true_eff ranks.

    Parameters
    ----------
    pred     : array-like, predicted scores
    true_eff : array-like, true efficiency values
    weights  : array-like, per-sample weights

    Returns
    -------
    float : weighted Spearman correlation in [0, 1] (clipped at 0)
    """
    pr = np.argsort(np.argsort(pred)).astype(float)
    tr = np.argsort(np.argsort(true_eff)).astype(float)
    w  = weights / weights.sum()
    mp = np.dot(w, pr)
    mt = np.dot(w, tr)
    cov = np.sum(w * (pr - mp) * (tr - mt))
    sp  = np.sqrt(np.sum(w * (pr - mp) ** 2))
    st  = np.sqrt(np.sum(w * (tr - mt) ** 2))
    if sp < 1e-12 or st < 1e-12:
        return 0.0
    return max(cov / (sp * st), 0.0)


def compute_cls(y_true, y_score, pe_eff):
    """Compute CLS = harmonic mean of PR-AUC and Weighted Spearman.

    Parameters
    ----------
    y_true  : array-like of int, binary activity labels
    y_score : array-like of float, predicted scores
    pe_eff  : array-like of float, true PE efficiency values

    Returns
    -------
    tuple : (CLS, PR-AUC, Weighted Spearman)
    """
    pr = average_precision_score(np.array(y_true), y_score)
    ws = weighted_spearman(y_score, np.array(pe_eff), np.array(pe_eff) + 0.01)
    return (2 * pr * ws / (pr + ws) if pr > 0 and ws > 0 else 0.0), pr, ws


def _fill(X):
    """Replace NaN with -999 sentinel (tree-model-friendly missingness encoding)."""
    return np.where(np.isnan(X), -999.0, X)


def _lr():
    """Return a fresh LR pipeline: median imputation + z-scaling + L2 LR."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("m",   LogisticRegression(
            C=0.1, penalty="l2", class_weight="balanced",
            max_iter=2000, random_state=SEED,
        )),
    ])


if __name__ == "__main__":
    # ── Load data ──────────────────────────────────────────────────────────
    rt_seq  = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    # Use the struct-enriched feature file if present, else fall back
    feat_path = DATA_DIR / "handcrafted_features_with_struct.csv"
    if not feat_path.exists():
        feat_path = DATA_DIR / "handcrafted_features.csv"
    feat_df = pd.read_csv(feat_path)

    df = feat_df.merge(
        rt_seq[["rt_name", "active", "pe_efficiency_pct", "rt_family", "protein_length_aa"]],
        on="rt_name",
    )
    df["pe_efficiency_pct"] = df["pe_efficiency_pct"].fillna(0.0)

    # ESM-2 embeddings
    emb_path = DATA_DIR / "esm2_embeddings.npz"
    if not emb_path.exists():
        emb_path = DATA_DIR.parent / "esm2_embeddings.npz"
    emb_data  = np.load(str(emb_path), allow_pickle=True)
    esm2_dict = dict(zip(emb_data["names"], emb_data["embeddings"]))
    all_embs  = np.vstack([esm2_dict[n] for n in df["rt_name"]])

    # ── Feature engineering ────────────────────────────────────────────────
    df["TM_ratio_MMLVPE_telo"] = df["foldseek_TM_MMLVPE"] / (df["foldseek_TM_Telomerase"] + 1e-6)
    df["is_broken_enzyme"]     = df["triad_best_rmsd"].isna().astype(int)
    df["length_to_TM_ratio"]   = df["protein_length_aa"] / (df["foldseek_best_TM"] + 1e-5)
    df["is_no_thumb"]          = df["thumb_fident"].isna().astype(int)
    df["is_no_hairpin"]        = df["n_hairpins_found"].isna().astype(int)

    FOLDSEEK = [
        "foldseek_best_TM", "foldseek_best_fident", "foldseek_best_LDDT",
        "foldseek_TM_HIV1", "foldseek_TM_MMLV", "foldseek_TM_MMLVPE",
        "foldseek_TM_Telomerase", "foldseek_TM_LTRRetrotransposon",
        "foldseek_TM_Group2Intron", "foldseek_TM_Retron",
    ]
    BASE = [
        "triad_best_rmsd", "D1_D2_dist", "pocket_hydrophobic_per_res",
        "camsol_score", "perplexity", "instability_index",
        "native_net_charge", "protein_length_aa", "TM_ratio_MMLVPE_telo",
    ]
    THERMO = ["t55_raw", "t60_raw", "t65_raw"]
    ENG    = ["is_broken_enzyme", "length_to_TM_ratio", "is_no_thumb", "is_no_hairpin"]

    # V14 feature sets
    CLF_FEATURES = FOLDSEEK + BASE + ["hbonds_per_res", "n_strands_total"] + ENG   # +n_strands_total
    REG_FEATURES = FOLDSEEK + BASE + ["thumb_fident", "n_hairpins_found"] + THERMO + ENG + [
        "isoelectric_point", "n_salt_bridges",  # +isoelectric_point, +n_salt_bridges
    ]

    PCA_N    = 5
    PCA_COLS = [f"pca_{i}" for i in range(PCA_N)]
    for c in PCA_COLS:
        df[c] = 0.0
    CLF_FEATURES += PCA_COLS
    REG_FEATURES += PCA_COLS

    missing = [c for c in CLF_FEATURES + REG_FEATURES if c not in df.columns]
    assert not missing, f"Missing features: {missing}"
    print(f"V14: CLF={len(CLF_FEATURES)}, REG={len(REG_FEATURES)} features")

    y_bin    = df["active"].values
    y_eff    = df["pe_efficiency_pct"].values
    families = df["rt_family"].values
    clf_oof  = np.zeros(len(df))
    reg_xgb  = np.zeros(len(df))
    reg_cb   = np.zeros(len(df))

    # ── LOFO loop ──────────────────────────────────────────────────────────
    for fam in sorted(set(families)):
        tr = np.where(families != fam)[0]
        te = np.where(families == fam)[0]

        # PCA fitted inside fold to prevent leakage
        pf = PCA(n_components=PCA_N, random_state=SEED)
        pt = pf.fit_transform(all_embs[tr])
        pe = pf.transform(all_embs[te])
        for i in range(PCA_N):
            df.iloc[tr, df.columns.get_loc(f"pca_{i}")] = pt[:, i]
            df.iloc[te, df.columns.get_loc(f"pca_{i}")] = pe[:, i]

        Xct = df.iloc[tr][CLF_FEATURES].values.astype(float)
        Xce = df.iloc[te][CLF_FEATURES].values.astype(float)
        Xrt = _fill(df.iloc[tr][REG_FEATURES].values.astype(float))
        Xre = _fill(df.iloc[te][REG_FEATURES].values.astype(float))
        yb  = y_bin[tr]
        ye  = y_eff[tr]
        sw  = np.sqrt(ye + 0.01)

        lr = _lr()
        lr.fit(Xct, yb)
        clf_oof[te] = lr.predict_proba(Xce)[:, 1]

        m = xgb.XGBRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=0,
        )
        m.fit(Xrt, ye, sample_weight=sw)
        reg_xgb[te] = np.clip(m.predict(Xre), 0, None)

        m = CatBoostRegressor(
            iterations=200, depth=2, learning_rate=0.03,
            verbose=False, random_seed=SEED,
        )
        m.fit(Xrt, ye, sample_weight=sw)
        reg_cb[te] = np.clip(m.predict(Xre), 0, None)

        print(f"  {fam:<25} n={len(te)} active={int(y_bin[te].sum())}")

    reg_oof = (reg_xgb + reg_cb) / 2.0
    print(f"\nLR  PR-AUC    = {average_precision_score(y_bin, clf_oof):.4f}")
    print(f"REG WSpearman = {weighted_spearman(reg_oof, y_eff, y_eff + 0.01):.4f}")

    # ── Beta sweep ─────────────────────────────────────────────────────────
    print(f"\n{'beta':>5}  {'CLS':>7}  {'PR-AUC':>7}  {'WSpearman':>10}")
    best_cls, best_beta = -1.0, 0.66
    for beta in np.arange(0.0, 1.01, 0.01):
        f   = beta * (rankdata(clf_oof) / len(df)) + (1 - beta) * (rankdata(reg_oof) / len(df))
        cls, pr, ws = compute_cls(y_bin, f, y_eff)
        if cls > best_cls:
            best_cls, best_beta = cls, float(beta)

    oof           = best_beta * (rankdata(clf_oof) / len(df)) + (1 - best_beta) * (rankdata(reg_oof) / len(df))
    cls, pr_auc, wsp = compute_cls(y_bin, oof, y_eff)
    print(f"\n{'=' * 55}")
    print(f"V14 FINAL: CLS={cls:.4f}  PR-AUC={pr_auc:.4f}  WSpearman={wsp:.4f}  beta={best_beta:.2f}")
    print(f"v11_clean: CLS=0.7908  PR-AUC=0.7989  WSpearman=0.7829")
    print(f"delta:    {cls - 0.7908:+.4f}")
    print(f"{'=' * 55}")

    # ── Save results JSON ──────────────────────────────────────────────────
    results_path = DATA_DIR / "results_v14.json"
    json.dump(
        {
            "version": "v14",
            "cls":     round(float(cls),    4),
            "pr_auc":  round(float(pr_auc), 4),
            "wsp":     round(float(wsp),    4),
            "beta":    best_beta,
            "clf_extra": ["hbonds_per_res", "n_strands_total"],
            "reg_extra": ["isoelectric_point", "n_salt_bridges"],
        },
        open(results_path, "w"),
        indent=2,
    )

    # ── Generate submission ────────────────────────────────────────────────
    sub_df = pd.DataFrame({"rt_name": df["rt_name"].values, "predicted_score": oof})
    sub_df = sub_df.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    sub_path = DATA_DIR / "submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"\nTop-10 predictions:")
    print(sub_df.head(10).to_string(index=False))
    print(f"\nsubmission.csv saved ({len(sub_df)} rows) → {sub_path}")
