import pandas as pd
from pathlib import Path

def load_partition(path):
    return pd.read_csv(path, sep="\t")

def load_all_data(data_dir):
    data_dir = Path(data_dir)

    files = [
        "sepsisexp_timeseries_partition-A.tsv",
        "sepsisexp_timeseries_partition-B.tsv",
        "sepsisexp_timeseries_partition-C.tsv",
        "sepsisexp_timeseries_partition-D.tsv"
    ]

    datasets = []
    for f in files:
        df = load_partition(data_dir / f)
        datasets.append(df)

    return datasets