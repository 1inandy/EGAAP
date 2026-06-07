"""Decision experiments on PERIOD-HOLDOUT (steps 1 & 2).

Step 2 (honest ceiling): the full 76-feature temporal XGBoost evaluated under
period-holdout (train on two periods, test on the third, rotate) -- the number
the report's headline should rest on.

Step 1 (partition question): does the solar-wind regime help more as a feature or
as a partition? A heuristic regime label (quiet / CIR high-speed-stream / CME
ejecta) is derived from the inputs, then:
  - baseline : 76-feature model, no regime          (== the step-2 ceiling)
  - A        : 76 features + regime one-hot          (regime-as-feature)
  - B        : per-regime specialist models, routed  (regime-as-partition)
Compared on AP (storm classification) and severe-storm MAE (Dst <= -100). If B
does not clearly beat A, partitioning is not worth building.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

AUX = ["smoothed_ssn", "gse_x_ace", "prop_delay_min"]
STORM = -50
SEVERE = -100
DATA_PATH = "hourly_features.csv"
OUT_DIR = "metrics"
REGIME_NAMES = {0: "quiet", 1: "CIR/HSS", 2: "CME"}


def derive_regime(df):
    """Heuristic solar-wind regime from inputs (no target leakage).

    CME/ICME ejecta: proton temperature far below that expected for the speed
    (cold plasma). CIR / high-speed stream: fast wind that is not ejecta. Else
    quiet. Expected-temperature relation after Lopez (1987)-style scaling.
    """
    V = df["speed"].values
    T = df["temperature"].values
    texp = (np.clip(0.031 * V - 5.1, 0, None)) ** 2 * 1e3
    cme = T < 0.5 * texp
    cir = (~cme) & (V > 500)
    return np.where(cme, 2, np.where(cir, 1, 0)).astype(int)


def make_reg():
    return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=8,
                        subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=42)


def make_clf(spw):
    return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=8,
                         min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
                         scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1,
                         random_state=42)


def metrics(y, pred_reg, proba):
    yc = (y <= STORM).astype(int)
    sev = y <= SEVERE
    return {
        "AP": average_precision_score(yc, proba),
        "AUC": roc_auc_score(yc, proba),
        "R2": r2_score(y, pred_reg),
        "MAE_all": mean_absolute_error(y, pred_reg),
        "MAE_sev": mean_absolute_error(y[sev], pred_reg[sev]) if sev.sum() else np.nan,
        "n_sev": int(sev.sum()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["timedelta"] = pd.to_timedelta(df["timedelta"])
    df = df.sort_values(["period", "timedelta"]).reset_index(drop=True)
    feats76 = [c for c in df.columns if c not in ("period", "timedelta", "dst")]
    df["regime"] = derive_regime(df)
    onehot = pd.get_dummies(df["regime"]).reindex(columns=[0, 1, 2], fill_value=0)
    onehot.columns = [f"regime_{i}" for i in (0, 1, 2)]
    df = pd.concat([df, onehot], axis=1)
    featsA = feats76 + list(onehot.columns)

    print("Regime mix (all data):")
    print(df["regime"].map(REGIME_NAMES).value_counts().to_string(), "\n")

    fold_rows = []
    for hold in ["train_a", "train_b", "train_c"]:
        te = (df["period"] == hold).values
        tr = ~te
        ytr, yte = df.loc[tr, "dst"].values, df.loc[te, "dst"].values
        regime_tr, regime_te = df.loc[tr, "regime"].values, df.loc[te, "regime"].values
        yc_tr = (ytr <= STORM).astype(int)
        spw = (yc_tr == 0).sum() / max((yc_tr == 1).sum(), 1)

        X76_tr, X76_te = df.loc[tr, feats76].values, df.loc[te, feats76].values
        XA_tr, XA_te = df.loc[tr, featsA].values, df.loc[te, featsA].values

        # baseline (step-2 ceiling) and A (regime-as-feature)
        base_reg = make_reg().fit(X76_tr, ytr)
        base_clf = make_clf(spw).fit(X76_tr, yc_tr)
        base_pred, base_proba = base_reg.predict(X76_te), base_clf.predict_proba(X76_te)[:, 1]

        A_reg = make_reg().fit(XA_tr, ytr)
        A_clf = make_clf(spw).fit(XA_tr, yc_tr)
        A_pred, A_proba = A_reg.predict(XA_te), A_clf.predict_proba(XA_te)[:, 1]

        # B (regime-as-partition); fall back to baseline for thin/degenerate regimes
        B_pred, B_proba = base_pred.copy(), base_proba.copy()
        for r in (0, 1, 2):
            tr_r, te_r = regime_tr == r, regime_te == r
            if te_r.sum() == 0 or tr_r.sum() < 500:
                continue
            B_pred[te_r] = make_reg().fit(X76_tr[tr_r], ytr[tr_r]).predict(X76_te[te_r])
            yc_r = (ytr[tr_r] <= STORM).astype(int)
            if yc_r.sum() >= 20 and yc_r.sum() < tr_r.sum():
                spw_r = (yc_r == 0).sum() / (yc_r == 1).sum()
                B_proba[te_r] = make_clf(spw_r).fit(
                    X76_tr[tr_r], yc_r).predict_proba(X76_te[te_r])[:, 1]

        for label, pred, proba in [("baseline", base_pred, base_proba),
                                   ("A: feature", A_pred, A_proba),
                                   ("B: partition", B_pred, B_proba)]:
            fold_rows.append({"fold": hold, "model": label, **metrics(yte, pred, proba)})

    table = pd.DataFrame(fold_rows).set_index(["model", "fold"])
    means = table.groupby("model").mean(numeric_only=True)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.width", 140)
    print("Per-fold (period-holdout):\n")
    print(table.to_string())
    print("\nMean across folds:\n")
    print(means.to_string())

    # Bar chart: mean AP and severe-MAE per model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    order = ["baseline", "A: feature", "B: partition"]
    colors = ["#6f7c86", "#56b6ab", "#c9924e"]
    ax1.bar(order, means.loc[order, "AP"], color=colors)
    ax1.set_title("Period-holdout AP (storm)"); ax1.set_ylabel("Average Precision")
    ax2.bar(order, means.loc[order, "MAE_sev"], color=colors)
    ax2.set_title("Severe-storm MAE (Dst<=-100)"); ax2.set_ylabel("MAE (nT)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/regime_compare.png", dpi=120); plt.close(fig)
    print(f"\nPlot written to ./{OUT_DIR}/regime_compare.png")


if __name__ == "__main__":
    main()
