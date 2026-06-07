"""Step 3 - attack the extreme-storm tail (Dst < -100).

The plain regressor compresses severe storms toward the mean. Three remedies are
compared on the leakage-free forward-chaining split with the full feature set,
using XGBoost (fast) throughout:

  1. baseline     : ordinary XGBRegressor
  2. weighted     : XGBRegressor with sample weights that upweight deep storms
  3. two-stage    : storm/quiet classifier gates a storm-specialist regressor
                    (trained only on storm hours) over the baseline regressor

Reported as error on the whole test set and on the tail subsets that matter.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

AUX = ["smoothed_ssn", "gse_x_ace", "prop_delay_min"]
STORM = -50
SEVERE = -100
DATA_PATH = "hourly_features.csv"
OUT_DIR = "metrics"


def forward_chain_mask(df):
    test = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("period").groups.items():
        idx = np.array(sorted(idx))
        test[idx[int(len(idx) * 0.8):]] = True
    return test


def subset_metrics(name, y, pred):
    out = {}
    for label, m in [("all", np.ones(len(y), bool)),
                     ("Dst<=-50", y <= STORM), ("Dst<=-100", y <= SEVERE)]:
        if m.sum() == 0:
            continue
        out[f"MAE_{label}"] = mean_absolute_error(y[m], pred[m])
        out[f"bias_{label}"] = float(np.mean(pred[m] - y[m]))  # +ve => over, -ve => under
    out["n_severe"] = int((y <= SEVERE).sum())
    return {"method": name, "R2": r2_score(y, pred), **out}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["timedelta"] = pd.to_timedelta(df["timedelta"])
    df = df.sort_values(["period", "timedelta"]).reset_index(drop=True)
    test = forward_chain_mask(df)
    feats = [c for c in df.columns if c not in ("period", "timedelta", "dst")]

    Xtr, Xte = df.loc[~test, feats].values, df.loc[test, feats].values
    ytr, yte = df.loc[~test, "dst"].values, df.loc[test, "dst"].values

    def make_reg():
        return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=8,
                            subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=42)

    rows = []

    # 1. Baseline
    base = make_reg().fit(Xtr, ytr)
    pred_base = base.predict(Xte)
    rows.append(subset_metrics("baseline", yte, pred_base))

    # 2. Sample-weighted (upweight deep storms): weight grows with storm depth
    w = 1.0 + np.clip(-ytr, 0, None) / 25.0   # Dst -100 -> 5x, -250 -> 11x
    pred_w = make_reg().fit(Xtr, ytr, sample_weight=w).predict(Xte)
    rows.append(subset_metrics("weighted", yte, pred_w))

    # 3. Two-stage: classifier gates a storm-specialist regressor
    yc = (ytr <= STORM).astype(int)
    spw = (yc == 0).sum() / (yc == 1).sum()
    gate = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=8,
                         min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
                         scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1,
                         random_state=42).fit(Xtr, yc)
    storm_mask_tr = ytr <= STORM
    specialist = make_reg().fit(Xtr[storm_mask_tr], ytr[storm_mask_tr])
    gate_proba = gate.predict_proba(Xte)[:, 1]
    pred_two = pred_base.copy()
    use_specialist = gate_proba >= 0.5
    pred_two[use_specialist] = specialist.predict(Xte[use_specialist])
    rows.append(subset_metrics("two-stage", yte, pred_two))

    table = pd.DataFrame(rows).set_index("method")
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.width", 140)
    print(f"Forward-chaining; full features ({len(feats)}); XGBoost.")
    print(f"test severe hours (Dst<=-100): {int((yte<=SEVERE).sum())}\n")
    print(table.to_string())

    # Plot: actual vs predicted on the storm region (Dst <= -50)
    sm = yte <= STORM
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(yte[sm], pred_base[sm], s=12, alpha=0.5, color="#a65a7e", label="baseline")
    ax.scatter(yte[sm], pred_two[sm], s=12, alpha=0.5, color="#56b6ab", label="two-stage")
    lo = yte[sm].min()
    ax.plot([lo, STORM], [lo, STORM], "--", color="#6f7c86", lw=1.5, label="perfect")
    ax.set_xlabel("Actual Dst (nT)"); ax.set_ylabel("Predicted Dst (nT)")
    ax.set_title("Storm-hour predictions (Dst <= -50)")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/tail_scatter.png", dpi=120); plt.close(fig)
    print(f"\nPlot written to ./{OUT_DIR}/tail_scatter.png")


if __name__ == "__main__":
    main()
