"""
retroviral_wall_visualisations.py
==================================
Standalone Python script that reproduces v14 OOF predictions via LOFO and
generates six publication-quality figures for the Mandrake Bioworks
Open Problems #1 — "The Retroviral Wall" challenge.

All data is loaded from the data/ directory relative to this script.
Figures are saved to figures/ at 300 DPI.

Usage:
    python retroviral_wall_visualisations.py
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from catboost import CatBoostRegressor

# ── Paths ──────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
DPI  = 300

# ── Style ──────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # fallback to default

# ── Feature definitions (mirror v14 exactly) ───────────────────────────────
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
NEW_STRUCT = ["thumb_fident", "n_hairpins_found", "isoelectric_point", "n_salt_bridges",
              "hbonds_per_res", "n_strands_total"]
PCA_N    = 5
PCA_COLS = [f"pca_{i}" for i in range(PCA_N)]

# Feature colour map (by category)
FEAT_COLORS = {}
for f in FOLDSEEK:
    FEAT_COLORS[f] = "orange"
for f in BASE + THERMO:
    FEAT_COLORS[f] = "steelblue"
for f in ENG:
    FEAT_COLORS[f] = "grey"
for f in NEW_STRUCT:
    FEAT_COLORS[f] = "#2ecc71"
for f in PCA_COLS:
    FEAT_COLORS[f] = "mediumpurple"
# Thermal stability override to red
for f in THERMO:
    FEAT_COLORS[f] = "#e74c3c"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load and merge all data sources, engineering the same features as v14."""
    rt_seq  = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    feat_df = pd.read_csv(DATA_DIR / "handcrafted_features_with_struct.csv")
    df = feat_df.merge(
        rt_seq[["rt_name", "active", "pe_efficiency_pct", "rt_family", "protein_length_aa"]],
        on="rt_name",
    )
    df["pe_efficiency_pct"] = df["pe_efficiency_pct"].fillna(0.0)

    # ESM-2 embeddings live one level up (large file, not duplicated into data/)
    esm_path = DATA_DIR.parent / "esm2_embeddings.npz"
    if not esm_path.exists():
        # If not found, create zero embeddings as fallback (disables ESM-2 PCA signal)
        print(f"WARNING: {esm_path} not found. Using zero ESM-2 embeddings.")
        n_emb = 1280
        esm2_dict = {row["rt_name"]: np.zeros(n_emb) for _, row in df.iterrows()}
    else:
        emb_data  = np.load(str(esm_path), allow_pickle=True)
        esm2_dict = dict(zip(emb_data["names"], emb_data["embeddings"]))

    all_embs = np.vstack([esm2_dict[n] for n in df["rt_name"]])

    # Engineered features
    df["TM_ratio_MMLVPE_telo"] = df["foldseek_TM_MMLVPE"] / (df["foldseek_TM_Telomerase"] + 1e-6)
    df["is_broken_enzyme"]     = df["triad_best_rmsd"].isna().astype(int)
    df["length_to_TM_ratio"]   = df["protein_length_aa"] / (df["foldseek_best_TM"] + 1e-5)
    df["is_no_thumb"]          = df["thumb_fident"].isna().astype(int)
    df["is_no_hairpin"]        = df["n_hairpins_found"].isna().astype(int)

    for c in PCA_COLS:
        df[c] = 0.0

    return df, all_embs


# ══════════════════════════════════════════════════════════════════════════════
# V14 LOFO LOOP (needed for feature importances, per-fold CLS, rank table)
# ══════════════════════════════════════════════════════════════════════════════

def run_lofo(df, all_embs):
    """
    Reproduce the v14 LOFO loop.

    Returns
    -------
    clf_oof  : np.ndarray, LR predicted probabilities (OOF)
    reg_oof  : np.ndarray, (XGB+CB)/2 predicted efficiency (OOF)
    oof      : np.ndarray, final blended rank score (OOF)
    best_beta: float
    xgb_importances: dict {feature: mean_gain}
    fold_results: list of dicts with per-fold CLS info
    """
    CLF_FEATURES = FOLDSEEK + BASE + ["hbonds_per_res", "n_strands_total"] + ENG + PCA_COLS
    REG_FEATURES = (FOLDSEEK + BASE + ["thumb_fident", "n_hairpins_found"] +
                    THERMO + ENG + ["isoelectric_point", "n_salt_bridges"] + PCA_COLS)

    y_bin    = df["active"].values
    y_eff    = df["pe_efficiency_pct"].values
    families = df["rt_family"].values

    clf_oof = np.zeros(len(df))
    reg_xgb = np.zeros(len(df))
    reg_cb  = np.zeros(len(df))

    xgb_imp_accum = {}
    fold_results  = []

    def _fill(X):
        return np.where(np.isnan(X), -999.0, X)

    def _lr():
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("m",   LogisticRegression(
                C=0.1, penalty="l2", class_weight="balanced",
                max_iter=2000, random_state=SEED,
            )),
        ])

    for fam in sorted(set(families)):
        tr = np.where(families != fam)[0]
        te = np.where(families == fam)[0]

        # PCA fitted inside fold (no leakage)
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

        # Classifier
        lr = _lr()
        lr.fit(Xct, yb)
        clf_oof[te] = lr.predict_proba(Xce)[:, 1]

        # XGBoost regressor
        m_xgb = xgb.XGBRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=0,
        )
        m_xgb.fit(Xrt, ye, sample_weight=sw)
        reg_xgb[te] = np.clip(m_xgb.predict(Xre), 0, None)

        # Accumulate XGBoost feature importance (gain)
        imp = m_xgb.get_booster().get_score(importance_type="gain")
        for k, v in imp.items():
            feat_name = REG_FEATURES[int(k[1:])]
            xgb_imp_accum[feat_name] = xgb_imp_accum.get(feat_name, 0.0) + v

        # CatBoost regressor
        m_cb = CatBoostRegressor(
            iterations=200, depth=2, learning_rate=0.03,
            verbose=False, random_seed=SEED,
        )
        m_cb.fit(Xrt, ye, sample_weight=sw)
        reg_cb[te] = np.clip(m_cb.predict(Xre), 0, None)

        print(f"  {fam:<25} n={len(te)} active={int(y_bin[te].sum())}")

    reg_oof = (reg_xgb + reg_cb) / 2.0

    # Beta sweep
    best_cls, best_beta = -1.0, 0.66
    beta_vals, cls_vals, pr_vals, ws_vals = [], [], [], []
    for beta in np.arange(0.0, 1.01, 0.01):
        f   = beta * (rankdata(clf_oof) / len(df)) + (1 - beta) * (rankdata(reg_oof) / len(df))
        cls, pr, ws = compute_cls(y_bin, f, y_eff)
        beta_vals.append(round(beta, 2))
        cls_vals.append(cls)
        pr_vals.append(pr)
        ws_vals.append(ws)
        if cls > best_cls:
            best_cls, best_beta = cls, float(beta)

    oof = best_beta * (rankdata(clf_oof) / len(df)) + (1 - best_beta) * (rankdata(reg_oof) / len(df))
    cls_final, pr_final, ws_final = compute_cls(y_bin, oof, y_eff)
    print(f"\nV14 FINAL: CLS={cls_final:.4f}  PR-AUC={pr_final:.4f}  WSpearman={ws_final:.4f}  beta={best_beta:.2f}")

    # Per-fold CLS
    for fam in sorted(set(families)):
        te = np.where(families == fam)[0]
        f_fold = best_beta * (rankdata(clf_oof) / len(df)) + (1 - best_beta) * (rankdata(reg_oof) / len(df))
        # Per-fold CLS (using only the fold's predictions vs the overall ranking)
        fold_oof = np.zeros(len(df))
        fold_oof[te] = oof[te]
        # Recompute CLS for this fold's held-out samples in context of all 57
        # Use global oof but compute CLS restricted to fold members
        fold_y_bin = y_bin[te]
        fold_y_eff = y_eff[te]
        fold_scores = oof[te]
        if len(te) < 2 or fold_y_bin.sum() == 0:
            fold_cls = np.nan
        else:
            try:
                fc, fp, fw = compute_cls(fold_y_bin, fold_scores, fold_y_eff)
                fold_cls = fc
            except Exception:
                fold_cls = np.nan
        fold_results.append({
            "family": fam,
            "n": len(te),
            "n_active": int(y_bin[te].sum()),
            "cls": fold_cls,
        })

    xgb_importances = {k: v / max(xgb_imp_accum.values()) for k, v in xgb_imp_accum.items()}

    return clf_oof, reg_oof, oof, best_beta, xgb_importances, fold_results, beta_vals, cls_vals, pr_vals, ws_vals


def weighted_spearman(pred, true_eff, weights):
    """Compute weighted Spearman correlation between pred ranks and true_eff ranks."""
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
    """CLS = harmonic mean of PR-AUC and Weighted Spearman."""
    pr = average_precision_score(np.array(y_true), y_score)
    ws = weighted_spearman(y_score, np.array(pe_eff), np.array(pe_eff) + 0.01)
    if pr > 0 and ws > 0:
        return (2 * pr * ws / (pr + ws)), pr, ws
    return 0.0, pr, ws


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — CLS Progression
# ══════════════════════════════════════════════════════════════════════════════

def fig_cls_progression():
    """Line chart showing CLS score progression across model versions (v9→v14)."""
    versions = ["v9", "v10", "v11", "v14"]
    scores   = [0.677, 0.7824, 0.7908, 0.8034]
    annotations = [
        "XGB/CB baseline",
        "LR clf + thumb_fident\n+ n_hairpins + PCA fix",
        "hbonds_per_res\nadded to CLF",
        "n_strands_total in CLF\nisoelectric_point +\nn_salt_bridges in REG\nbeta=0.66",
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(versions, scores, "o-", color="#2980b9", lw=2.5, markersize=10, zorder=3)
    ax.axhline(scores[0], color="#e74c3c", lw=1.4, ls="--", alpha=0.7, label=f"v9 baseline (CLS={scores[0]:.3f})")

    # Colour the markers
    colors = ["#95a5a6", "#27ae60", "#f39c12", "#8e44ad"]
    for i, (v, s, col) in enumerate(zip(versions, scores, colors)):
        ax.scatter([v], [s], color=col, s=120, zorder=4, edgecolors="k", linewidths=0.8)

    # Annotations with arrows
    offsets = [(0, 0.022), (0, 0.022), (0, -0.032), (0, 0.022)]
    for i, (v, s, ann, off) in enumerate(zip(versions, scores, annotations, offsets)):
        ax.annotate(
            ann,
            xy=(v, s),
            xytext=(v, s + off[1]),
            ha="center", va="bottom" if off[1] > 0 else "top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.15, edgecolor=colors[i]),
            arrowprops=dict(arrowstyle="-", color=colors[i], lw=0.8),
        )

    # Delta labels between steps
    for i in range(1, len(versions)):
        delta = scores[i] - scores[i - 1]
        mid_x = i - 0.5
        mid_y = (scores[i] + scores[i - 1]) / 2
        ax.text(mid_x, mid_y + 0.004, f"+{delta:.4f}", ha="center", fontsize=9,
                color="#2c3e50", fontweight="bold")

    ax.set_xlabel("Model Version", fontsize=12)
    ax.set_ylabel("CLS Score (harmonic mean of PR-AUC and Weighted Spearman)", fontsize=10)
    ax.set_title("CLS Score Progression Across Model Versions", fontsize=13, fontweight="bold")
    ax.set_ylim(0.62, 0.87)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))

    plt.tight_layout()
    out = FIG_DIR / "cls_progression.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

def fig_feature_importance(xgb_importances):
    """Horizontal bar chart of top 15 XGBoost features coloured by category."""
    # Sort and take top 15
    sorted_imp = sorted(xgb_importances.items(), key=lambda x: x[1], reverse=True)[:15]
    feats, imps = zip(*sorted_imp)

    colors = [FEAT_COLORS.get(f, "steelblue") for f in feats]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(range(len(feats)), imps, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised XGBoost Gain (averaged over LOFO folds)", fontsize=10)
    ax.set_title("Top 15 Feature Importances by Category (XGBoost Regressor)", fontsize=12, fontweight="bold")

    # Legend
    legend_items = [
        mpatches.Patch(color="orange",       label="FoldSeek / Structural TM"),
        mpatches.Patch(color="steelblue",    label="Biophysical"),
        mpatches.Patch(color="#e74c3c",      label="Thermal Stability"),
        mpatches.Patch(color="grey",         label="Engineered Flags"),
        mpatches.Patch(color="#2ecc71",      label="New Structural (thumb/hairpin/pI)"),
        mpatches.Patch(color="mediumpurple", label="ESM-2 PCA"),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="lower right")
    plt.tight_layout()
    out = FIG_DIR / "feature_importance.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Feature Distributions (Active vs Inactive)
# ══════════════════════════════════════════════════════════════════════════════

def fig_feature_distributions(df):
    """2x3 violin + strip plots for top 6 features, split by active/inactive."""
    top_features = [
        "foldseek_best_TM", "thumb_fident", "n_hairpins_found",
        "hbonds_per_res", "isoelectric_point", "camsol_score",
    ]
    feat_labels = {
        "foldseek_best_TM":    "FoldSeek Best TM-score",
        "thumb_fident":        "Thumb Subdomain Identity\n(ρ=0.901 with PE efficiency)",
        "n_hairpins_found":    "N Hairpins Found\n(ρ=0.816 with PE efficiency)",
        "hbonds_per_res":      "H-bonds per Residue",
        "isoelectric_point":   "Isoelectric Point (pI)",
        "camsol_score":        "CamSol Solubility Score",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    palette = {"Active": "#2ecc71", "Inactive": "#e74c3c"}

    for ax, feat in zip(axes, top_features):
        plot_df = df[["active", feat]].copy().dropna(subset=[feat])
        plot_df["Label"] = plot_df["active"].map({1: "Active", 0: "Inactive"})

        sns.violinplot(
            data=plot_df, x="Label", y=feat,
            palette=palette, inner=None, cut=0, alpha=0.55, ax=ax, order=["Active", "Inactive"],
        )
        sns.stripplot(
            data=plot_df, x="Label", y=feat,
            palette=palette, size=5, jitter=True,
            edgecolor="k", linewidth=0.5, ax=ax, order=["Active", "Inactive"],
            alpha=0.85,
        )
        ax.set_title(feat_labels[feat], fontsize=9.5, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(feat, fontsize=8)

        # Show n
        for i, label in enumerate(["Active", "Inactive"]):
            n = (plot_df["Label"] == label).sum()
            ax.text(i, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom", fontsize=8, color="grey")

    fig.suptitle("Feature Distributions: Active vs Inactive RTs", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "feature_distributions.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Family Overview
# ══════════════════════════════════════════════════════════════════════════════

def fig_family_overview(df):
    """Horizontal stacked bar chart of RT families with active/inactive split."""
    fam_counts = (
        df.groupby("rt_family")
        .agg(n_active=("active", "sum"), n_inactive=("active", lambda x: (x == 0).sum()))
        .reset_index()
    )
    fam_counts["total"] = fam_counts["n_active"] + fam_counts["n_inactive"]
    fam_counts = fam_counts.sort_values("n_active", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(fam_counts))

    bars_act = ax.barh(y, fam_counts["n_active"],   color="#2ecc71", edgecolor="k", linewidth=0.6, label="Active")
    bars_ina = ax.barh(y, fam_counts["n_inactive"], left=fam_counts["n_active"],
                       color="#e74c3c", edgecolor="k", linewidth=0.6, label="Inactive")

    # Annotate total
    for i, row in fam_counts.iterrows():
        ax.text(row["total"] + 0.15, i, f"n={row['total']}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(fam_counts["rt_family"], fontsize=10)
    ax.set_xlabel("Number of RTs", fontsize=11)
    ax.set_title("RT Family Composition (Active vs Inactive)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, fam_counts["total"].max() + 2)

    # Highlight Retroviral
    retro_idx = fam_counts.index[fam_counts["rt_family"] == "Retroviral"].tolist()
    if retro_idx:
        ax.get_yticklabels()[retro_idx[0]].set_fontweight("bold")
        ax.get_yticklabels()[retro_idx[0]].set_color("#2980b9")

    plt.tight_layout()
    out = FIG_DIR / "family_overview.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Rank Ensemble Table
# ══════════════════════════════════════════════════════════════════════════════

def fig_rank_ensemble(df, clf_oof, reg_oof, oof, best_beta):
    """
    Visual table showing rank blending for a representative sample of ~8 RTs.
    Rows coloured by true label.
    """
    y_bin = df["active"].values
    y_eff = df["pe_efficiency_pct"].values
    n     = len(df)

    lr_rank  = rankdata(clf_oof) / n
    reg_rank = rankdata(reg_oof) / n
    blend    = best_beta * lr_rank + (1 - best_beta) * reg_rank
    final_rk = rankdata(blend)

    # Pick ~8 representative RTs: top 4 + bottom 2 + 2 interesting middle
    sorted_idx = np.argsort(final_rk)[::-1]
    chosen = list(sorted_idx[:4]) + list(sorted_idx[-2:]) + [sorted_idx[28], sorted_idx[35]]
    chosen = list(dict.fromkeys(chosen))[:8]

    rows = []
    for idx in chosen:
        rows.append({
            "RT Name":     df.iloc[idx]["rt_name"],
            "True Label":  "Active" if y_bin[idx] else "Inactive",
            "LR Prob":     f"{clf_oof[idx]:.3f}",
            "LR Rank":     f"{lr_rank[idx]:.3f}",
            "Reg Score":   f"{reg_oof[idx]:.3f}",
            "Reg Rank":    f"{reg_rank[idx]:.3f}",
            f"Blend\n({best_beta:.2f}×LR + {1-best_beta:.2f}×Reg)": f"{blend[idx]:.3f}",
            "Final Rank":  f"{int(final_rk[idx]):d}",
        })

    tab_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")

    col_labels = list(tab_df.columns)
    cell_text  = tab_df.values.tolist()

    row_colors = []
    for row in rows:
        if row["True Label"] == "Active":
            row_colors.append(["#d5f5e3"] * len(col_labels))
        else:
            row_colors.append(["#fadbd8"] * len(col_labels))

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=row_colors,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_fontsize(8)

    ax.set_title(
        f"Rank-Based Blend: Combining Classifier and Regressor  (beta={best_beta:.2f})\n"
        "Ranks remove calibration bias between LR and XGB/CB   |   "
        "Green rows = Active, Red rows = Inactive",
        fontsize=10, fontweight="bold", pad=14,
    )

    plt.tight_layout()
    out = FIG_DIR / "rank_ensemble.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — LOFO Per-Fold CLS
# ══════════════════════════════════════════════════════════════════════════════

def fig_lofo_performance(fold_results):
    """Bar chart of per-fold CLS with overall dashed line."""
    # Filter folds that have CLS (folds with at least 1 active)
    plot_folds = [f for f in fold_results if not np.isnan(f["cls"])]
    families   = [f["family"] for f in plot_folds]
    cls_vals   = [f["cls"]    for f in plot_folds]

    retroviral_families = {"Retroviral"}
    colors = ["#e74c3c" if fam in retroviral_families else "steelblue" for fam in families]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(families))
    bars = ax.bar(x, cls_vals, color=colors, edgecolor="k", linewidth=0.6, alpha=0.85)

    ax.axhline(0.8034, color="#2c3e50", lw=1.8, ls="--", label="Overall CLS = 0.8034")

    # Value labels
    for bar, val in zip(bars, cls_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # n info
    for i, f in enumerate(plot_folds):
        ax.text(i, 0.02, f"n={f['n']}\n({f['n_active']} active)",
                ha="center", va="bottom", fontsize=7, color="white" if colors[i] == "#e74c3c" else "k")

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Per-Fold CLS Score", fontsize=11)
    ax.set_title("LOFO Per-Fold CLS Performance", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)

    # Legend patches
    legend_patches = [
        mpatches.Patch(color="#e74c3c", label="Retroviral"),
        mpatches.Patch(color="steelblue", label="Other families"),
    ]
    ax.legend(handles=legend_patches + [plt.Line2D([0], [0], color="#2c3e50", lw=1.8, ls="--",
                                                    label="Overall CLS = 0.8034")],
              fontsize=9)

    # Note for folds with no actives (CLS undefined)
    skipped = [f["family"] for f in fold_results if np.isnan(f["cls"])]
    if skipped:
        ax.text(0.99, 0.97, f"Skipped (0 actives): {', '.join(skipped)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="grey",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    out = FIG_DIR / "lofo_performance.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Retroviral Wall Visualisations — v14")
    print("=" * 60)

    print("\n[1/3] Loading data...")
    df, all_embs = load_data()
    print(f"Loaded {len(df)} RTs | Active: {df['active'].sum()} | Families: {df['rt_family'].nunique()}")

    print("\n[2/3] Running v14 LOFO loop (this may take ~60s)...")
    clf_oof, reg_oof, oof, best_beta, xgb_importances, fold_results, beta_vals, cls_vals, pr_vals, ws_vals = \
        run_lofo(df, all_embs)

    print("\n[3/3] Generating figures...")

    print("\n  Fig 1: CLS Progression")
    fig_cls_progression()

    print("  Fig 2: Feature Importance")
    fig_feature_importance(xgb_importances)

    print("  Fig 3: Feature Distributions")
    fig_feature_distributions(df)

    print("  Fig 4: Family Overview")
    fig_family_overview(df)

    print("  Fig 5: Rank Ensemble Table")
    fig_rank_ensemble(df, clf_oof, reg_oof, oof, best_beta)

    print("  Fig 6: LOFO Per-Fold CLS")
    fig_lofo_performance(fold_results)

    print("\n" + "=" * 60)
    print("All figures saved to:", FIG_DIR)
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  {p.name}")
    print("=" * 60)
