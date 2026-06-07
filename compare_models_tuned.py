"""Tuned classifier comparison for the storm event (Dst <= -50 nT):

  - Logistic Regression (class_weight='balanced')
  - Linear SVM / LinearSVC (class_weight='balanced')
  - RBF SVM, GridSearchCV-tuned over (C, gamma), class_weight='balanced'
  - Random Forest
  - XGBoost (scale_pos_weight for imbalance)

Scale-sensitive models run inside a StandardScaler pipeline. The RBF SVM is
tuned/trained on a stratified subsample (kernel SVMs are O(n^2)); everything is
evaluated on the full held-out test set. Models without predict_proba are scored
via decision_function (fine for ranking metrics / curves).
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

FEATURES = ["speed", "bt", "temperature", "bz_gsm", "density"]
TARGET = "dst"
DATA_PATH = "hourly_avg_dst.csv"
STORM_THRESHOLD = -50
OUT_DIR = "metrics"
SVM_SUBSAMPLE = 12000

COLORS = {
    "LogReg (balanced)": "#56b6ab",
    "Linear SVM (balanced)": "#c9924e",
    "RBF SVM (tuned)": "#a65a7e",
    "Random Forest": "#5a8fc9",
    "XGBoost": "#8ec96b",
}


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)[FEATURES + [TARGET]].dropna()
    X = df[FEATURES]
    y = (df[TARGET] <= STORM_THRESHOLD).astype(int).values

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                  stratify=y)
    X_train, _, y_train, _ = train_test_split(X_tv, y_tv, test_size=0.2, random_state=42,
                                               stratify=y_tv)
    X_svm, _, y_svm, _ = train_test_split(X_train, y_train, train_size=SVM_SUBSAMPLE,
                                          random_state=42, stratify=y_train)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    base_rate = y_test.mean()
    print(f"Test storm base rate: {base_rate*100:.2f}%  ({y_test.sum():,}/{len(y_test):,})")
    print(f"scale_pos_weight (neg/pos) = {pos_weight:.2f}")
    print(f"RBF SVM tuned on {len(X_svm):,} rows; all models tested on {len(X_test):,}\n")

    # Grid-search the RBF SVM.
    svm_grid = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
        param_grid={"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.1, 1]},
        scoring="average_precision", cv=3, n_jobs=-1,
    )
    svm_grid.fit(X_svm, y_svm)
    print(f"Best RBF SVM params: {svm_grid.best_params_} "
          f"(CV AP={svm_grid.best_score_:.3f})\n")

    models = {
        "LogReg (balanced)": (
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000, class_weight="balanced")),
            X_train, y_train),
        "Linear SVM (balanced)": (
            make_pipeline(StandardScaler(),
                          LinearSVC(class_weight="balanced", max_iter=5000, dual="auto")),
            X_train, y_train),
        "RBF SVM (tuned)": (svm_grid.best_estimator_, X_svm, y_svm),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            X_train, y_train),
        "XGBoost": (
            XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6,
                          subsample=0.9, colsample_bytree=0.9,
                          scale_pos_weight=pos_weight, eval_metric="logloss",
                          n_jobs=-1, random_state=42),
            X_train, y_train),
    }

    scores, rows = {}, []
    for name, (model, Xtr, ytr) in models.items():
        if name != "RBF SVM (tuned)":  # already fitted via grid search
            model.fit(Xtr, ytr)
        proba = get_scores(model, X_test)
        scores[name] = proba
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "ROC_AUC": roc_auc_score(y_test, proba),
            "AvgPrec": average_precision_score(y_test, proba),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        })

    table = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(table.to_string())

    def overlay(curve_fn, xlabel, ylabel, title, fname, ref=None):
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for name, proba in scores.items():
            x, y_, lab = curve_fn(name, proba)
            ax.plot(x, y_, color=COLORS[name], lw=2, label=lab)
        if ref:
            ref(ax)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout(); fig.savefig(f"{OUT_DIR}/{fname}", dpi=120); plt.close(fig)

    overlay(
        lambda n, p: (*roc_curve(y_test, p)[:2], f"{n} ({roc_auc_score(y_test, p):.3f})"),
        "False Positive Rate", "True Positive Rate", "ROC - tuned comparison",
        "tuned_roc.png", ref=lambda ax: ax.plot([0, 1], [0, 1], "--", color="#6f7c86", lw=1))

    def pr(n, p):
        prec, rec, _ = precision_recall_curve(y_test, p)
        return rec, prec, f"{n} (AP={average_precision_score(y_test, p):.3f})"
    overlay(pr, "Recall", "Precision", "Precision-Recall - tuned comparison",
            "tuned_pr.png",
            ref=lambda ax: ax.axhline(base_rate, ls="--", color="#6f7c86", lw=1))

    def gains(n, p):
        ys = y_test[np.argsort(-p)]
        pop = np.arange(1, len(ys) + 1) / len(ys)
        return pop, np.cumsum(ys) / ys.sum(), n
    overlay(gains, "Fraction targeted", "Fraction of storms captured",
            "Cumulative Gains - tuned comparison", "tuned_gains.png",
            ref=lambda ax: ax.plot([0, 1], [0, 1], "--", color="#6f7c86", lw=1))

    print(f"\nPlots written to ./{OUT_DIR}/ (tuned_roc, tuned_pr, tuned_gains)")


if __name__ == "__main__":
    main()
