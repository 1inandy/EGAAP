"""Feature engineering for Dst prediction.

Builds temporal features (rolling stats, lags, gradients, integrated southward
Bz) plus the previously-unused auxiliary inputs (sunspot number, L1 propagation
delay) on top of the hourly base table. Everything is computed STRICTLY within
each period so no window ever crosses a period boundary, and only past/current
information feeds each row (no look-ahead).

Input:  hourly_avg_dst.csv  (period, timedelta, 5 base features, dst)
Output: hourly_features.csv  (period, timedelta, dst, base + engineered features)
"""

import numpy as np
import pandas as pd

BASE = ["speed", "bt", "temperature", "bz_gsm", "density"]
ROLL_WINDOWS = [3, 6, 12, 24]
LAGS = [1, 3, 6]
BZ_INT_WINDOWS = [6, 12, 24]
WARMUP = 24  # drop the first 24 h of each period so all windows/lags are valid

IN_PATH = "hourly_avg_dst.csv"
OUT_PATH = "hourly_features.csv"


def add_temporal(df):
    df = df.sort_values(["period", "timedelta"]).reset_index(drop=True)
    g = df.groupby("period", group_keys=False)

    for f in BASE:
        for w in ROLL_WINDOWS:
            df[f"{f}_rmean{w}"] = g[f].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            df[f"{f}_rstd{w}"] = g[f].transform(lambda s, w=w: s.rolling(w, min_periods=2).std())
        for lag in LAGS:
            df[f"{f}_lag{lag}"] = g[f].shift(lag)
        df[f"{f}_grad1"] = g[f].diff(1)
        df[f"{f}_grad3"] = g[f].diff(3)

    # Integrated southward Bz: accumulated energy-input proxy over recent hours.
    df["bz_south"] = (-df["bz_gsm"]).clip(lower=0)
    gs = df.groupby("period", group_keys=False)
    for w in BZ_INT_WINDOWS:
        df[f"bz_south_int{w}"] = gs["bz_south"].transform(
            lambda s, w=w: s.rolling(w, min_periods=1).sum()
        )
    return df


def add_sunspots(df):
    ss = pd.read_csv("sunspots.csv")
    ss["timedelta"] = pd.to_timedelta(ss["timedelta"])
    out = []
    for period, grp in df.groupby("period", sort=False):
        sp = ss[ss["period"] == period].sort_values("timedelta")
        merged = pd.merge_asof(
            grp.sort_values("timedelta"), sp[["timedelta", "smoothed_ssn"]],
            on="timedelta", direction="backward",
        )
        out.append(merged)
    df = pd.concat(out, ignore_index=True)
    df["smoothed_ssn"] = df.groupby("period")["smoothed_ssn"].bfill()
    return df


def add_propagation_delay(df):
    """L1 -> Earth advection delay (minutes) from ACE x-position and wind speed."""
    sp = pd.read_csv("satellite_pos.csv", usecols=["period", "timedelta", "gse_x_ace"])
    sp["timedelta"] = pd.to_timedelta(sp["timedelta"])
    out = []
    for period, grp in df.groupby("period", sort=False):
        s = sp[sp["period"] == period].sort_values("timedelta")
        merged = pd.merge_asof(
            grp.sort_values("timedelta"), s[["timedelta", "gse_x_ace"]],
            on="timedelta", direction="backward",
        )
        out.append(merged)
    df = pd.concat(out, ignore_index=True)
    df["gse_x_ace"] = df.groupby("period")["gse_x_ace"].bfill()
    # delay (min) = distance (km) / speed (km/s) / 60; guard against zero speed.
    df["prop_delay_min"] = df["gse_x_ace"] / df["speed"].replace(0, np.nan) / 60.0
    return df


def main():
    df = pd.read_csv(IN_PATH)
    df["timedelta"] = pd.to_timedelta(df["timedelta"])
    df = df.dropna(subset=["dst"])

    df = add_temporal(df)
    df = add_sunspots(df)
    df = add_propagation_delay(df)

    # Drop per-period warmup so lag/rolling features are fully defined.
    df = df.sort_values(["period", "timedelta"]).reset_index(drop=True)
    df = df[df.groupby("period").cumcount() >= WARMUP].reset_index(drop=True)
    df = df.drop(columns=["bz_south"])

    # Any residual gaps (e.g. early sunspot/position) -> period median then 0.
    feat_cols = [c for c in df.columns if c not in ("period", "timedelta", "dst")]
    df[feat_cols] = df.groupby("period")[feat_cols].transform(lambda s: s.fillna(s.median()))
    df[feat_cols] = df[feat_cols].fillna(0)

    df.to_csv(OUT_PATH, index=False)
    print(f"{OUT_PATH}: {len(df):,} rows, {len(feat_cols)} features "
          f"(base {len(BASE)} + engineered {len(feat_cols) - len(BASE)})")
    print("remaining NaNs:", int(df[feat_cols].isna().sum().sum()))


if __name__ == "__main__":
    main()
