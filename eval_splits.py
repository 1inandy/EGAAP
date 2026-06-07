"""Step 1 - quantify validation leakage from random splitting.

Same base-feature Random Forest, evaluated under three split strategies:

  - random          : stratified 80/20 (the ORIGINAL, leaky protocol)
  - period-holdout  : train on two periods, test on the third (rotate all three)
  - forward-chaining: first 80% of each period -> train, last 20% -> test

Periods are independent series and `timedelta` restarts per period, so random
splitting lets near-duplicate adjacent hours straddle train/test. The drop from
'random' to the time-aware splits is the leakage we are removing.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE = ["speed", "bt", "temperature", "bz_gsm", "density"]
STORM_THRESHOLD = -50
DATA_PATH = "hourly_avg_dst.csv"


def fit_eval(X_tr, y_tr, X_te, y_te):
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(X_tr, y_tr)
    p = reg.predict(X_te)
    reg_metrics = dict(R2=r2_score(y_te, p), MAE=mean_absolute_error(y_te, p),
                       RMSE=mean_squared_error(y_te, p) ** 0.5)

    yc_tr = (y_tr <= STORM_THRESHOLD).astype(int)
    yc_te = (y_te <= STORM_THRESHOLD).astype(int)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_tr, yc_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    clf_metrics = dict(AUC=roc_auc_score(yc_te, proba),
                       AP=average_precision_score(yc_te, proba),
                       storm_rate=yc_te.mean())
    return reg_metrics, clf_metrics


def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["dst"])
    df["timedelta"] = pd.to_timedelta(df["timedelta"])
    df = df.sort_values(["period", "timedelta"]).reset_index(drop=True)
    X, y = df[BASE], df["dst"]

    rows = []

    # Random (original, leaky)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=(y <= STORM_THRESHOLD))
    r, c = fit_eval(Xtr, ytr, Xte, yte)
    rows.append({"split": "random (leaky)", **r, **c})

    # Period-holdout (rotate)
    for hold in ["train_a", "train_b", "train_c"]:
        te = df["period"] == hold
        r, c = fit_eval(X[~te], y[~te], X[te], y[te])
        rows.append({"split": f"period-holdout: test {hold}", **r, **c})

    # Forward-chaining within each period
    test_mask = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("period").groups.items():
        idx = np.array(sorted(idx))
        cut = int(len(idx) * 0.8)
        test_mask[idx[cut:]] = True
    r, c = fit_eval(X[~test_mask], y[~test_mask], X[test_mask], y[test_mask])
    rows.append({"split": "forward-chaining", **r, **c})

    table = pd.DataFrame(rows).set_index("split")
    # Mean of the three period-holdout folds.
    ph = table[table.index.str.startswith("period-holdout")]
    table.loc["period-holdout: MEAN"] = ph.mean(numeric_only=True)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 120)
    print(f"Base features only ({len(BASE)}). RandomForest. n={len(df):,}\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
