import pandas as pd
from etna.datasets.tsdataset import TSDataset


def convert_to_ts_df(df: pd.DataFrame) -> TSDataset:
    """Convert pandas DataFrane to TSDataset."""
    ts_df = df.copy()
    ts_df["timestamp"] = ts_df["date"]
    ts_df["segment"] = ts_df["country"] + "_" + ts_df["store"] + "_" + ts_df["product"]
    ts_df["target"] = ts_df["num_sold"]

    ts_df = ts_df[["timestamp", "segment", "target"]]
    ts_df = TSDataset(ts_df, freq="D")

    return ts_df
