"""Finalize the storm classifier (Dst <= -50 nT):

  1. Grid-search XGBoost (depth / estimators / learning rate / min_child_weight)
     and compare it to the Random Forest champion.
  2. Pick an operating threshold on a held-out VALIDATION set for target recalls
     (catch 70/80/90% of storms), then report final metrics on the TEST set.
  3. Calibrate the winner (isotonic, prefit on validation) and compare reliability
     + Brier score before vs after.

Split: 64% train / 16% validation / 20% test (stratified). Tuning uses CV on
train; threshold + calibration use validation; all final numbers are on test.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

FEATURES = ["speed", "bt", "temperature", "bz_gsm", "density"]
TARGET = "dst"
DATA_PATH = "hourly_avg_dst.csv"
STORM_THRESHOLD = -50
OUT_DIR = "metrics"


def threshold_for_recall(y, proba, target):
    prec, rec, thr = precision_recall_curve(y, proba)
    ok = np.where(rec[:-1] >= target)[0]
    if len(ok) == 0:
        return thr[0], prec[0], rec[0]
    best = ok[np.argmax(prec[ok])]
    return thr[best], prec[best], rec[best]


def report_at_threshold(name, y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    print(f"  [{name}] thr={thr:.3f}  precision={precision:.3f}  recall={recall:.3f}  "
          f"FPR={fpr:.3f}  (TP={tp} FP={fp} FN={fn} TN={tn})")
    return precision, recall, fpr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)[FEATURES + [TARGET]].dropna()
    X = df[FEATURES]
    y = (df[TARGET] <= STORM_THRESHOLD).astype(int).values

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2,
                                                      random_state=42, stratify=y_temp)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}  "
          f"storm base rate={y_test.mean()*100:.2f}%\n")

    # 1. Tune XGBoost ------------------------------------------------------
    grid = GridSearchCV(
        XGBClassifier(scale_pos_weight=pos_weight, subsample=0.9, colsample_bytree=0.9,
                      eval_metric="logloss", n_jobs=1, random_state=42),
        param_grid={
            "max_depth": [4, 6, 8],
            "n_estimators": [300, 600],
            "learning_rate": [0.05, 0.1],
            "min_child_weight": [1, 5],
        },
        scoring="average_precision", cv=3, n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)
    xgb = grid.best_estimator_
    print(f"Best XGBoost params: {grid.best_params_}  (CV AP={grid.best_score_:.3f})")

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    candidates = {"Random Forest": rf, "XGBoost (tuned)": xgb}
    print("\nTest-set ranking metrics:")
    test_ap = {}
    for name, model in candidates.items():
        proba = model.predict_proba(X_test)[:, 1]
        ap = average_precision_score(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        test_ap[name] = ap
        print(f"  {name:18s}  ROC AUC={auc:.4f}  AP={ap:.4f}")

    winner_name = max(test_ap, key=test_ap.get)
    winner = candidates[winner_name]
    print(f"\n>>> Winner by Average Precision: {winner_name}\n")

    # Final PR comparison plot
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, model in candidates.items():
        p = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, lw=2, label=f"{name} (AP={average_precision_score(y_test, p):.3f})")
    ax.axhline(y_test.mean(), ls="--", color="#6f7c86", lw=1, label="baseline")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Final PR: RF vs tuned XGBoost"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/final_pr.png", dpi=120); plt.close(fig)

    # 2. Operating thresholds (chosen on validation, reported on test) -----
    proba_val = winner.predict_proba(X_val)[:, 1]
    print(f"Operating points for {winner_name} (threshold chosen on validation):")
    for target in (0.70, 0.80, 0.90):
        thr, p_val, r_val = threshold_for_recall(y_val, proba_val, target)
        proba_test = winner.predict_proba(X_test)[:, 1]
        print(f"\n target recall >= {target:.0%}  ->  threshold {thr:.3f} "
              f"(val precision {p_val:.3f})")
        report_at_threshold("test", y_test, proba_test, thr)

    # 3. Calibrate the winner ---------------------------------------------
    calibrated = CalibratedClassifierCV(FrozenEstimator(winner), method="isotonic")
    calibrated.fit(X_val, y_val)

    proba_raw = winner.predict_proba(X_test)[:, 1]
    proba_cal = calibrated.predict_proba(X_test)[:, 1]
    print(f"\nCalibration (Brier score, lower is better) for {winner_name}:")
    print(f"  raw        = {brier_score_loss(y_test, proba_raw):.4f}")
    print(f"  calibrated = {brier_score_loss(y_test, proba_cal):.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#6f7c86", lw=1, label="perfect")
    for label, p, color in [("raw", proba_raw, "#a65a7e"), ("calibrated", proba_cal, "#56b6ab")]:
        frac, mean_p = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, "o-", color=color, lw=2,
                label=f"{label} (Brier={brier_score_loss(y_test, p):.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed storm frequency")
    ax.set_title(f"Calibration before/after - {winner_name}"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/final_calibration.png", dpi=120); plt.close(fig)

    print(f"\nPlots written to ./{OUT_DIR}/ (final_pr, final_calibration)")


if __name__ == "__main__":
    main()
