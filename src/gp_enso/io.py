from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

@dataclass
class PreparedData:
    df: pd.DataFrame
    t: pd.Series

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    scaler_x: StandardScaler
    scaler_y: StandardScaler

    X_train_scaled: np.ndarray
    y_train_scaled: np.ndarray
    X_test_scaled: np.ndarray


def load_noaa_index(index_name: str) -> pd.DataFrame:
    """
    Load a NOAA monthly index into a tidy DataFrame with columns:
      - date (datetime64[ns], month start)
      - <index_name> (float)

    Robust to junk rows / missing-value sentinels like -99.99.
    """
    url = (
        f"https://psl.noaa.gov/data/timeseries/month/data/{index_name}.long.anom.data"
        if index_name.startswith("nino")
        else f"https://psl.noaa.gov/data/correlation/{index_name}.data"
    )

    raw = pd.read_csv(url, sep=r"\s+", header=None, skiprows=1)

    # 1) Year column can contain junk like "-99.99" in some files -> coerce, then drop invalid rows
    year = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    mask = year.notna() & year.between(1800, 3000)  # generous sanity bounds
    raw = raw.loc[mask].copy()
    years = year.loc[mask].astype(int).to_numpy()

    # 2) Coerce the 12 monthly columns to numeric (strings -> floats, junk -> NaN)
    months_df = raw.iloc[:, 1:13].apply(pd.to_numeric, errors="coerce")

    # 3) Build dates by (year, month) so values stay aligned even if some years are missing
    years_rep = np.repeat(years, 12)
    months_rep = np.tile(np.arange(1, 13), len(years))
    dates = pd.to_datetime({"year": years_rep, "month": months_rep, "day": 1})

    values = months_df.to_numpy().ravel(order="C")
    s = pd.Series(values, index=dates)

    # 4) Drop NOAA missing-value sentinel and NaNs
    s = s.where(s.ne(-99.99)).dropna()

    return s.rename(index_name).rename_axis("date").reset_index()

def build_df(
    indices: Iterable[str] = ("nino3", "soi", "nino4", "nino34", "nino12"),
    how: str = "inner",
) -> pd.DataFrame:
    """
    Load a set of NOAA indices and merge them on date into one DataFrame.
    """
    dfs = [load_noaa_index(name) for name in indices]
    df_new = reduce(lambda left, right: left.merge(right, on="date", how=how), dfs)
    df_new = df_new.sort_values("date", ignore_index=True)
    return df_new

def add_time_years(df: pd.DataFrame, date_col: str = "date", out_col: str = "t_years") -> pd.DataFrame:
    """
    Adds a float column 't_years' = years since start (approx, using 365 days).
    """
    df = df.copy()
    df[out_col] = (df[date_col] - df[date_col].min()).dt.days / 365.0
    return df

def prepare_train_test(
    df: pd.DataFrame,
    *,
    target: str = "nino34",
    features: tuple[str, ...] = ("t_years", "soi"),
    n_train: int = 700,
    split_at: int = 600,         # where the plot/test segment begins
    date_col: str = "date",
) -> PreparedData:
    """
    Creates t_years, builds X/y, fits StandardScalers on the first n_train rows,
    and returns scaled train and test arrays (test begins at split_at).
    """
    df2 = add_time_years(df, date_col=date_col, out_col="t_years")

    # Build full design matrix / target
    X_all = df2.loc[:, features].to_numpy(dtype=float)
    y_all = df2.loc[:, target].to_numpy(dtype=float)

    # Fit scalers on first n_train
    scaler_x = StandardScaler().fit(X_all[:n_train])
    scaler_y = StandardScaler().fit(y_all[:n_train].reshape(-1, 1))

    X_train = X_all[:n_train]
    y_train = y_all[:n_train]

    X_train_scaled = scaler_x.transform(X_train)
    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel()

    X_test = X_all[split_at:]
    y_test = y_all[split_at:]
    X_test_scaled = scaler_x.transform(X_test)

    return PreparedData(
        df=df2,
        t=df2["t_years"],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        X_train_scaled=X_train_scaled,
        y_train_scaled=y_train_scaled,
        X_test_scaled=X_test_scaled,
    )