"""Train the geomagnetic (Dst) RandomForest model and save it as geomagnetic_model.pkl.

Mirrors the modelling cells of the notebook: it uses the merged hourly dataset
(hourly_avg_dst.csv) and the same five solar-wind features, in the same order
that app.py feeds them to the model. If the merged dataset is missing, it is
rebuilt from the raw downloads via prepare_data.py.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import prepare_data

# Order matters: app.py builds the feature vector in exactly this order.
FEATURES = ["speed", "bt", "temperature", "bz_gsm", "density"]
TARGET = "dst"
DATA_PATH = "hourly_avg_dst.csv"
MODEL_PATH = "geomagnetic_model.pkl"


def load_dataset() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    print(f"{DATA_PATH} not found - rebuilding from raw data...")
    return prepare_data.main()


def main() -> None:
    df = load_dataset()[FEATURES + [TARGET]].dropna()
    X, y = df[FEATURES], df[TARGET]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    for name, Xs, ys in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        pred = model.predict(Xs)
        print(f"{name} MSE: {mean_squared_error(ys, pred):.4f}  R2: {r2_score(ys, pred):.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
