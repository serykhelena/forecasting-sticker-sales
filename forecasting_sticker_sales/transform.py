import numpy as np
import pandas as pd
from etna.datasets.tsdataset import TSDataset


def convert_to_ts_df(df: pd.DataFrame) -> TSDataset:
    """Convert pandas DataFrane to TSDataset."""
    ts_df = df.copy()
    ts_df["timestamp"] = ts_df["date"]
    ts_df["segment"] = ts_df["country"] + "_" + ts_df["store"] + "_" + ts_df["product"]
    ts_df["target"] = ts_df["num_sold"]

    ts_df = ts_df[["id", "timestamp", "segment", "target"]]
    ts_df = TSDataset(ts_df, freq="D")

    return ts_df


def convert_to_df(ts_df: TSDataset) -> pd.DataFrame:
    """Convert TSDataset to pandas DataFrame."""
    df = ts_df.to_pandas(flatten=True)
    df = df.rename(columns={"timestamp": "date", "target": "num_sold"})
    df["country"] = df["segment"].apply(lambda x: x.split("_")[0])
    df["store"] = df["segment"].apply(lambda x: x.split("_")[1])
    df["product"] = df["segment"].apply(lambda x: x.split("_")[2])

    # round values after outlier processing
    df["num_sold"] = np.ceil(df["num_sold"].values)

    return df[["id", "date", "country", "store", "product", "num_sold"]]
