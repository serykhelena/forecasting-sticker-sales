import numpy as np
import pandas as pd
from prophet.make_holidays import make_holidays_df


def get_holiday_df(country: str) -> pd.DataFrame:
    """Get holiday information for the specific country."""
    holidays_df = make_holidays_df(year_list=list(range(2010, 2017)), country=country)

    holidays_df["holiday"] = 1
    holidays_df = holidays_df.rename(columns={"ds": "date"})
    holidays_df = holidays_df.set_index("date")

    return holidays_df


def create_dtime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create day-time features."""
    df_new = df.copy()

    df_new["dayofweek"] = df_new.index.dayofweek.tolist()
    df_new["cos_weekday"] = np.cos(df_new["dayofweek"] / 7 * 2 * np.pi)
    df_new["sin_weekday"] = np.sin(df_new["dayofweek"] / 7 * 2 * np.pi)

    df_new["is_leap_year"] = df_new.index.is_leap_year.tolist()
    df_new["dayofyear"] = df_new.index.dayofyear.tolist()
    df_new["cos_doy"] = np.cos(df_new["dayofyear"] / (365 + df_new["is_leap_year"]) * 2 * np.pi)
    df_new["sin_doy"] = np.sin(df_new["dayofyear"] / (365 + df_new["is_leap_year"]) * 2 * np.pi)

    df_new["dayofmonth"] = df_new.index.hour.tolist()
    df_new["quarter"] = df_new.index.quarter.tolist()
    df_new["year"] = df_new.index.year.tolist()
    df_new["month"] = df_new.index.month.tolist()
    df_new["day"] = df_new.index.day.tolist()
    df_new["weekofyear"] = df_new.index.isocalendar().week.tolist()

    df_new["is_weekend"] = 0
    df_new.loc[df_new["dayofweek"].isin([5, 6]), "is_weekend"] = 1

    df_new["is_month_start"] = df_new.index.is_month_start.tolist()
    df_new["is_month_end"] = df_new.index.is_month_end.tolist()

    return df_new


def create_holiday_features(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame:
    """Create holiday features."""
    df_new = df.copy()
    df_new = df_new.merge(holidays_df, left_index=True, right_index=True, how="outer")
    df_new["holiday"] = df_new["holiday"].fillna(0)
    df_new["holiday_lag_1"] = df_new["holiday"].shift(1)
    df_new["holiday_lag_1"] = df_new["holiday"].shift(2)
    df_new = df_new.dropna(subset=["num_sold"])

    return df_new


def create_lags(df: pd.DataFrame, horizon: int, lags_count: int, feature_name: str) -> pd.DataFrame:
    """Create lags features."""
    df_new = df.copy()
    for lag in range(horizon, horizon + lags_count):
        df_new[f"lag_{lags_count}"] = df_new[feature_name].shift(lag)

    return df_new


def create_rolling_mean(df: pd.DataFrame, feature_name: str, window: int) -> pd.DataFrame:
    """Create rolling mean features."""
    df_new = df.copy()
    df_new[f"{feature_name}_rolling_mean_{window}"] = df_new[feature_name].rolling(window).mean()

    return df_new


def create_features(df: pd.DataFrame, holidays_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create regression features."""
    feature_df = df.copy()

    feature_df = create_dtime_features(feature_df)
    feature_df = create_holiday_features(feature_df, holidays_df)

    lags = [7, 30, 60]
    for lag in lags:
        feature_df = create_lags(
            feature_df,
            horizon=horizon,
            lags_count=lag,
            feature_name="num_sold",
        )
        feature_df = create_rolling_mean(feature_df, f"lag_{lag}", 30)
        feature_df = create_rolling_mean(feature_df, f"lag_{lag}", 14)

    feature_df = feature_df.dropna()

    # column casting
    for col in feature_df.columns:
        if "rolling" in col:
            feature_df[col] = feature_df[col].astype(np.float32)
            continue

        feature_df[col] = feature_df[col].astype(int)

    return feature_df
