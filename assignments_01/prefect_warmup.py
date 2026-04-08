import prefect
import task, flow
import pandas as pd
import numpy as np

@task
def create_series(arr):
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]}

@flow
def pipeline_flow():
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    s = create_series(arr)
    c = clean_data(s)
    summary = summarize_data(c)

    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary

if __name__ == "__main__":
    pipeline_flow()

#This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?
    # Because it is so simple and used for such small tasks
#Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
    # If it stays simple but is used on a larger scale.