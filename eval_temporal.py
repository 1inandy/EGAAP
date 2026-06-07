"""Steps 2 & 4 - do temporal + auxiliary features help, on the leakage-free split?

Forward-chaining split (first 80% of each period -> train, last 20% -> test).
Three feature sets compared for both Random Forest and XGBoost, on regression
(Dst) and storm classification (Dst <= -50 nT):

  - base            : the 5 raw solar-wind features
  - base+temporal   : + rolling stats / lags / gradients / integrated southward Bz
  - base+temporal+aux : + sunspot number and L1 propagation delay (step 4)
"""

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

BASE = ["speed", "bt", "temperature", "bz_gsm", "density"]
AUX = ["smoothed_ssn", "gse_x_ace", "prop_delay_min"]
STORM_THRESHOLD = -50
DATA_PATH = "hourly_features.csv"
OUT_DIR = "metrics"


def forward_chain_mask(df):
    test = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("period").groups.items():
        idx = np.array(sorted(idx))
        test[idx[int(len(idx) * 0.8):]] = True
    return test


def rf_reg():
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


def rf_clf():
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


def xgb_reg():
    return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=8,
                        subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=42)


def xgb_clf(spw):
    return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=8,
                         min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
                         scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1,
                         random_state=42)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["timedelta"] = pd.to_timedelta(df["timedelta"])
    df = df.sort_values(["period", "timedelta"]).reset_index(drop=True)
    test = forward_chain_mask(df)

    all_feats = [c for c in df.columns if c not in ("period", "timedelta", "dst")]
    temporal = [c for c in all_feats if c not in AUX]
    feature_sets = {
        "base": BASE,
        "base+temporal": temporal,
        "base+temporal+aux": all_feats,
    }

    y = df["dst"].values
    ytr_reg, yte_reg = y[~test], y[test]
    yc = (df["dst"].values <= STORM_THRESHOLD).astype(int)
    ytr_c, yte_c = yc[~test], yc[test]
    spw = (ytr_c == 0).sum() / (ytr_c == 1).sum()
    print(f"n_train={int((~test).sum()):,}  n_test={int(test.sum()):,}  "
          f"test storm rate={yte_c.mean()*100:.2f}%\n")

    pr_curves = {}
    rows = []
    for fs_name, cols in feature_sets.items():
        Xtr, Xte = df.loc[~test, cols].values, df.loc[test, cols].values
        for model_name, reg, clf in [
            ("RandomForest", rf_reg(), rf_clf()),
            ("XGBoost", xgb_reg(), xgb_clf(spw)),
        ]:
            reg.fit(Xtr, ytr_reg)
            pr = reg.predict(Xte)
            clf.fit(Xtr, ytr_c)
            proba = clf.predict_proba(Xte)[:, 1]
            rows.append({
                "features": fs_name, "model": model_name, "n_feat": len(cols),
                "R2": r2_score(yte_reg, pr), "MAE": mean_absolute_error(yte_reg, pr),
                "RMSE": mean_squared_error(yte_reg, pr) ** 0.5,
                "AUC": roc_auc_score(yte_c, proba),
                "AP": average_precision_score(yte_c, proba),
            })
            pr_curves[(fs_name, model_name)] = proba
            print(f"  done: {fs_name:18s} {model_name}")

    table = pd.DataFrame(rows).set_index(["features", "model"])
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 120)
    print("\nForward-chaining split results:\n")
    print(table.to_string())

    # PR curves: base vs full for each model
    fig, ax = plt.subplots(figsize=(6.5, 5))
    styles = {("base", "RandomForest"): ("#a65a7e", "-"),
              ("base+temporal+aux", "RandomForest"): ("#56b6ab", "-"),
              ("base", "XGBoost"): ("#c9924e", "--"),
              ("base+temporal+aux", "XGBoost"): ("#5a8fc9", "--")}
    for key, (color, ls) in styles.items():
        proba = pr_curves[key]
        prec, rec, _ = precision_recall_curve(yte_c, proba)
        ap = average_precision_score(yte_c, proba)
        ax.plot(rec, prec, color=color, ls=ls, lw=2,
                label=f"{key[1]} / {key[0]} (AP={ap:.3f})")
    ax.axhline(yte_c.mean(), ls=":", color="#6f7c86", lw=1, label="baseline")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Temporal features - PR (forward-chaining)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/temporal_pr.png", dpi=120); plt.close(fig)
    print(f"\nPlot written to ./{OUT_DIR}/temporal_pr.png")


if __name__ == "__main__":
    main()
