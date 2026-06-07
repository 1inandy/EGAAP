"""Rebuild the cleaned/merged training table from the raw MagNet downloads.

Raw inputs (re)downloaded into the project root:
  - solar_wind.csv : minute-cadence solar-wind measurements, per period
  - labels.csv     : hourly Dst index, per period

`timedelta` restarts at 0 for each `period` (train_a/b/c are independent time
series), so everything is grouped by (period, timedelta) -- NOT timedelta alone,
which would incorrectly average the three periods together.

Outputs:
  - hourly_avg.csv     : solar-wind features averaged to the hour
  - hourly_avg_dst.csv : hourly features joined to the Dst target
"""

import pandas as pd

SOLAR_WIND_PATH = "solar_wind.csv"
LABELS_PATH = "labels.csv"
HOURLY_AVG_PATH = "hourly_avg.csv"
HOURLY_AVG_DST_PATH = "hourly_avg_dst.csv"

# Only the columns the model actually uses (keeps memory/time reasonable on the
# full ~8.4M-row file). Add more here if you want a richer feature set.
FEATURES = ["speed", "bt", "temperature", "bz_gsm", "density"]
KEYS = ["period", "timedelta"]


def build_hourly_average() -> pd.DataFrame:
    df = pd.read_csv(SOLAR_WIND_PATH, usecols=KEYS + FEATURES)
    df["timedelta"] = pd.to_timedelta(df["timedelta"]).dt.floor("h")
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    # Mean per (period, hour); mean() skips NaNs so partial gaps are handled.
    hourly = df.groupby(KEYS, as_index=False)[FEATURES].mean()
    hourly.to_csv(HOURLY_AVG_PATH, index=False)
    print(f"{HOURLY_AVG_PATH}: {len(hourly):,} hourly rows")
    return hourly


def merge_with_labels(hourly: pd.DataFrame) -> pd.DataFrame:
    labels = pd.read_csv(LABELS_PATH, usecols=KEYS + ["dst"])
    labels["timedelta"] = pd.to_timedelta(labels["timedelta"]).dt.floor("h")
    labels = labels.drop_duplicates(subset=KEYS, keep="first")

    merged = hourly.merge(labels, on=KEYS, how="left")
    merged.to_csv(HOURLY_AVG_DST_PATH, index=False)
    matched = merged["dst"].notna().mean() * 100
    print(f"{HOURLY_AVG_DST_PATH}: {len(merged):,} rows, {matched:.1f}% with a Dst label")
    return merged


def main() -> pd.DataFrame:
    hourly = build_hourly_average()
    return merge_with_labels(hourly)


if __name__ == "__main__":
    main()
