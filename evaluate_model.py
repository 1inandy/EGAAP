"""Full evaluation of the geomagnetic Dst model.

Two views of performance, written to the metrics/ folder as PNGs plus a printed
summary:

  A. Regression (the deployed model that app.py serves): MAE / RMSE / R2,
     predicted-vs-actual, residuals.
  B. Storm-event classification: the standard "moderate or stronger" storm event
     (Dst <= -50 nT). A probabilistic classifier is trained on the same split to
     produce confusion matrix, ROC + AUC, precision-recall + AP, calibration,
     lift, and cumulative-gains charts.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

FEATURES = ["speed", "bt", "temperature", "bz_gsm", "density"]
TARGET = "dst"
DATA_PATH = "hourly_avg_dst.csv"
MODEL_PATH = "geomagnetic_model.pkl"
STORM_THRESHOLD = -50  # Dst <= -50 nT == moderate (G2) or stronger storm.
OUT_DIR = "metrics"

ACCENT = "#56b6ab"
WARN = "#c9924e"
ALT = "#a65a7e"


def split(df):
    """Reproduce train_model.py's split exactly so 'test' is truly held out."""
    X, y = df[FEATURES], df[TARGET]
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, _, y_train, _ = train_test_split(X_tv, y_tv, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate_regression(model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print("\n=== A. REGRESSION (deployed model) ===")
    print(f"n_test = {len(y_test):,}")
    print(f"MAE  = {mae:.3f} nT")
    print(f"RMSE = {rmse:.3f} nT")
    print(f"R^2  = {r2:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, s=6, alpha=0.25, color=ACCENT, edgecolors="none")
    lo, hi = y_test.min(), y_test.max()
    ax.plot([lo, hi], [lo, hi], "--", color=WARN, lw=1.5, label="perfect")
    ax.set_xlabel("Actual Dst (nT)")
    ax.set_ylabel("Predicted Dst (nT)")
    ax.set_title(f"Predicted vs Actual Dst (R$^2$={r2:.3f})")
    ax.legend()
    save(fig, "regression_pred_vs_actual.png")

    resid = y_test.values - pred
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(resid, bins=80, color=ACCENT, alpha=0.85)
    ax.axvline(0, color=WARN, lw=1.5)
    ax.set_xlabel("Residual (actual - predicted, nT)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals (MAE={mae:.2f}, RMSE={rmse:.2f} nT)")
    save(fig, "regression_residuals.png")


def evaluate_classification(X_train, X_test, y_train_reg, y_test_reg):
    y_train = (y_train_reg <= STORM_THRESHOLD).astype(int).values
    y_test = (y_test_reg <= STORM_THRESHOLD).astype(int).values
    base_rate = y_test.mean()

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    print(f"\n=== B. STORM CLASSIFICATION (Dst <= {STORM_THRESHOLD} nT) ===")
    print(f"positive (storm) base rate in test = {base_rate*100:.2f}%  "
          f"({y_test.sum():,}/{len(y_test):,})")
    print(f"ROC AUC          = {auc:.4f}")
    print(f"Average Precision = {ap:.4f}")
    print("\nClassification report @ threshold 0.5:")
    print(classification_report(y_test, pred, target_names=["quiet", "storm"], digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion matrix @0.5  ->  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["quiet", "storm"]).plot(
        ax=ax, cmap="cividis", colorbar=False, values_format="d"
    )
    ax.set_title("Confusion Matrix (threshold = 0.5)")
    save(fig, "confusion_matrix.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#6f7c86", lw=1, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    save(fig, "roc_curve.png")

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color=ALT, lw=2, label=f"PR (AP={ap:.3f})")
    ax.axhline(base_rate, ls="--", color="#6f7c86", lw=1,
               label=f"baseline ({base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    save(fig, "pr_curve.png")

    # Calibration / reliability diagram
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="#6f7c86", lw=1, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color=ACCENT, lw=2, label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed storm frequency")
    ax.set_title("Calibration (Reliability Diagram)")
    ax.legend(loc="upper left")
    save(fig, "calibration.png")

    # Lift & cumulative gains
    order = np.argsort(-proba)
    y_sorted = y_test[order]
    n = len(y_sorted)
    pop_frac = np.arange(1, n + 1) / n
    cum_pos = np.cumsum(y_sorted)
    gains = cum_pos / y_sorted.sum()
    lift = gains / pop_frac

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pop_frac, gains, color=ACCENT, lw=2, label="model")
    ax.plot([0, 1], [0, 1], "--", color="#6f7c86", lw=1, label="random")
    perfect_x = min(base_rate, 1.0)
    ax.plot([0, perfect_x, 1], [0, 1, 1], ":", color=WARN, lw=1.2, label="perfect")
    ax.set_xlabel("Fraction of population targeted (highest risk first)")
    ax.set_ylabel("Fraction of storms captured")
    ax.set_title("Cumulative Gains")
    ax.legend(loc="lower right")
    save(fig, "cumulative_gains.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pop_frac, lift, color=ALT, lw=2, label="model")
    ax.axhline(1.0, ls="--", color="#6f7c86", lw=1, label="random (lift=1)")
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Lift (vs random)")
    ax.set_title("Lift Curve")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    save(fig, "lift_curve.png")

    for d in (0.1, 0.2):
        idx = int(d * n) - 1
        print(f"Top {int(d*100)}% targeted -> captures {gains[idx]*100:.1f}% of storms "
              f"(lift {lift[idx]:.2f}x)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)[FEATURES + [TARGET]].dropna()
    X_train, X_test, y_train, y_test = split(df)

    model = joblib.load(MODEL_PATH)
    evaluate_regression(model, X_test, y_test)
    evaluate_classification(X_train, X_test, y_train, y_test)
    print(f"\nPlots written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
