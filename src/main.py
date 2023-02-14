"""Main module for running the anomaly detection model."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from oats.models import RegressionModel
from oats.threshold import QuantileThreshold


def plot_anomaly(data, scores, percentile=0.99):
    threshold = QuantileThreshold()
    threshold_values = threshold.get_threshold(scores, percentile)
    anomalies = scores > threshold_values
    anomaly_idx = np.where(anomalies == True)

    fig, (data_sp, anom_scores_sp) = plt.subplots(2)
    data_sp.plot(data)
    anom_scores_sp.plot(pd.DataFrame(scores, index=data.index))
    anom_scores_sp.plot(pd.DataFrame(threshold_values, index=data.index), "-")
    for anom in anomaly_idx:
        data_sp.plot(data.index[anom], data[anom], "ro")

    return fig


if __name__ == '__main__':
    home = Path.home()
    data_path = home.joinpath("data")
    data_path.mkdir(exist_ok=True)
    print(f"Data path is: {data_path}")
    df = pd.read_parquet(data_path / "Production.parquet")

    farm_id = "24"

    df_farm = df[df["farm_id"] == farm_id]
    df_farm.index = pd.to_datetime(df_farm.index)
    df_farm = df_farm.asfreq(freq='D')
    df_farm_avg_milk_yield = df_farm["AvgYieldPerCowFromDailyMilk"]
    df_farm_avg_milk_yield.interpolate(inplace=True)

    TRAIN_SIZE = int(len(df_farm_avg_milk_yield) * 0.8)
    train, test = df_farm_avg_milk_yield[:TRAIN_SIZE], df_farm_avg_milk_yield[TRAIN_SIZE:]

    r_model = RegressionModel(window=10)
    r_model.fit(train.values)
    r_scores = r_model.get_scores(test.values)
    anomaly_image = plot_anomaly(test, r_scores, percentile=0.99)
    anomaly_image.savefig(data_path / "anomaly.png")