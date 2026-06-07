<<<<<<< HEAD
[solar_wind.csv](https://drive.google.com/file/d/1kb1oa8q9HKvXOLotmwH7iLxvDuzzsJyc/view?usp=sharing) 
[geomagnetic_model.pkl](https://drive.google.com/file/d/10vzz1pTYzZKWQ--HO8JyxLmBh7Wv73s0/view?usp=sharing)
These files are too big for Github, so please download it from here.



To run the app, please run  ```python app.py``` in the terminal
=======
# Geomagnetic Storm Predictor

A small Flask workbench for estimating the **Dst geomagnetic storm index** from
hourly solar-wind conditions. It supports one-off predictions, realistic scenario
fill-ins, CSV batch runs, and storm severity labels.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

The trained model file (`geomagnetic_model.pkl`) is large and not checked in.
Generate it from the raw data:

```bash
python train_model.py
```

If the merged dataset (`hourly_avg_dst.csv`) is missing, `train_model.py` rebuilds
it from the raw `solar_wind.csv` + `labels.csv` via `prepare_data.py`, then trains
and writes `geomagnetic_model.pkl`. You can also run the data step on its own:

```bash
python prepare_data.py   # solar_wind.csv + labels.csv -> hourly_avg[_dst].csv
```

Note: `timedelta` restarts per `period` (train_a/b/c are separate time series), so
the data is aggregated by `(period, timedelta)` rather than by `timedelta` alone.

## Run the app

```bash
python app.py
```

Then open http://127.0.0.1:5000.

The app expects these five inputs:

```csv
speed,bt,temperature,bz_gsm,density
420,8,90000,-4,6
```

CSV batch uploads use the same column names. The batch response reports how many
rows were predicted and how many were skipped because of missing or invalid data.

## Model performance

A full technical report — background, data, methodology, metric definitions,
model comparisons (Logistic Regression / SVM / Random Forest / XGBoost),
operating thresholds, calibration, and all graphs — is in
**[RESULTS.md](RESULTS.md)**.

Headline numbers (honest, time-aware evaluation of the temporal model):

- Forward-chaining split: **R² ≈ 0.67**, MAE ≈ 6.7 nT, storm AUC ≈ 0.98, AP ≈ 0.71
- Strictest unseen-epoch (period-holdout) ceiling: **R² ≈ 0.59, AP ≈ 0.63**
- A random split inflates these (R² 0.44 / AP 0.48); temporal features are the
  biggest accuracy lever, and partitioning by regime gave no benefit. Full
  analysis and graphs in [RESULTS.md](RESULTS.md).

Regenerate everything:

```bash
# Initial analysis (random split)
python evaluate_model.py        # regression + classification diagnostics
python compare_models.py        # LogReg vs SVM vs Random Forest
python compare_models_tuned.py  # + tuning, class balancing, XGBoost
python tune_and_finalize.py     # XGBoost tuning, thresholds, calibration

# Leakage correction & time-aware modeling
python prepare_features.py      # temporal + sunspot + propagation features
python eval_splits.py           # random vs period-holdout vs forward-chaining
python eval_temporal.py         # base vs temporal vs +auxiliary features
python eval_tail.py             # extreme-storm tail remedies
```

## Data

Raw inputs (from the DrivenData MagNet dataset), too large for GitHub:

- `solar_wind.csv` — minute-cadence solar-wind measurements (the main model input)
- `labels.csv` — hourly Dst target
- `satellite_pos.csv`, `sunspots.csv` — additional context, not yet used by the model

`prepare_data.py` derives `hourly_avg.csv` and `hourly_avg_dst.csv` from these.

- solar_wind.csv: https://drive.google.com/file/d/1kb1oa8q9HKvXOLotmwH7iLxvDuzzsJyc/view?usp=sharing
>>>>>>> 0fa6bfe (initial)
