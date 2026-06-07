"""Compare classifiers for the storm event (Dst <= -50 nT):
Logistic Regression, RBF-kernel SVM, and Random Forest.

Logistic regression and SVM are scale-sensitive, so they run inside a
StandardScaler pipeline. A full RBF SVM on ~90k rows is impractical (O(n^2)),
so the SVM is trained on a stratified subsample; every model is *evaluated* on
the full held-out test set for a fair comparison.

Outputs overlay ROC / PR / gains / lift charts in metrics/ plus a printed table.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = ["speed", "bt", "temperature", "bz_gsm", "density"]
TARGET = "dst"
DATA_PATH = "hourly_avg_dst.csv"
STORM_THRESHOLD = -50
OUT_DIR = "metrics"
SVM_SUBSAMPLE = 15000  # training rows for the RBF SVM (full set is too slow)

COLORS = {"Logistic Regression": "#56b6ab", "RBF SVM": "#c9924e", "Random Forest": "#a65a7e"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)[FEATURES + [TARGET]].dropna()
    X, y_reg = df[FEATURES], df[TARGET]
    y = (y_reg <= STORM_THRESHOLD).astype(int).values

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                  stratify=y)
    X_train, _, y_train, _ = train_test_split(X_tv, y_tv, test_size=0.2, random_state=42,
                                               stratify=y_tv)

    # Stratified subsample for the SVM only.
    if len(X_train) > SVM_SUBSAMPLE:
        X_svm, _, y_svm, _ = train_test_split(
            X_train, y_train, train_size=SVM_SUBSAMPLE, random_state=42, stratify=y_train
        )
    else:
        X_svm, y_svm = X_train, y_train

    models = {
        "Logistic Regression": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            X_train, y_train,
        ),
        "RBF SVM": (
            make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True,
                                                random_state=42)),
            X_svm, y_svm,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            X_train, y_train,
        ),
    }

    base_rate = y_test.mean()
    print(f"Test storm base rate: {base_rate*100:.2f}%  ({y_test.sum():,}/{len(y_test):,})")
    if len(X_svm) < len(X_train):
        print(f"(RBF SVM trained on {len(X_svm):,} of {len(X_train):,} rows)\n")

    scores, rows = {}, []
    for name, (model, Xtr, ytr) in models.items():
        model.fit(Xtr, ytr)
        proba = model.predict_proba(X_test)[:, 1]
        scores[name] = proba
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "ROC_AUC": roc_auc_score(y_test, proba),
            "AvgPrec": average_precision_score(y_test, proba),
            "precision@0.5": precision_score(y_test, pred, zero_division=0),
            "recall@0.5": recall_score(y_test, pred, zero_division=0),
            "f1@0.5": f1_score(y_test, pred, zero_division=0),
        })

    table = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(table.to_string())

    # Overlay ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in scores.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, color=COLORS[name], lw=2,
                label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#6f7c86", lw=1, label="random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC - model comparison"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/compare_roc.png", dpi=120); plt.close(fig)

    # Overlay PR
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in scores.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec, prec, color=COLORS[name], lw=2,
                label=f"{name} (AP={average_precision_score(y_test, proba):.3f})")
    ax.axhline(base_rate, ls="--", color="#6f7c86", lw=1, label=f"baseline ({base_rate:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall - model comparison"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/compare_pr.png", dpi=120); plt.close(fig)

    # Overlay cumulative gains & lift
    fig_g, ax_g = plt.subplots(figsize=(6, 5))
    fig_l, ax_l = plt.subplots(figsize=(6, 5))
    for name, proba in scores.items():
        order = np.argsort(-proba)
        ys = y_test[order]
        n = len(ys)
        pop = np.arange(1, n + 1) / n
        gains = np.cumsum(ys) / ys.sum()
        ax_g.plot(pop, gains, color=COLORS[name], lw=2, label=name)
        ax_l.plot(pop, gains / pop, color=COLORS[name], lw=2, label=name)
    ax_g.plot([0, 1], [0, 1], "--", color="#6f7c86", lw=1, label="random")
    ax_g.set_xlabel("Fraction targeted"); ax_g.set_ylabel("Fraction of storms captured")
    ax_g.set_title("Cumulative Gains - comparison"); ax_g.legend(loc="lower right")
    fig_g.tight_layout(); fig_g.savefig(f"{OUT_DIR}/compare_gains.png", dpi=120); plt.close(fig_g)
    ax_l.axhline(1, ls="--", color="#6f7c86", lw=1, label="random")
    ax_l.set_xlabel("Fraction targeted"); ax_l.set_ylabel("Lift"); ax_l.set_ylim(bottom=0)
    ax_l.set_title("Lift - comparison"); ax_l.legend(loc="upper right")
    fig_l.tight_layout(); fig_l.savefig(f"{OUT_DIR}/compare_lift.png", dpi=120); plt.close(fig_l)

    print(f"\nComparison plots written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
