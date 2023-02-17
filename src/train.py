"""Training script for the model."""
import datetime
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from oats.models import RegressionModel
from oats.threshold import QuantileThreshold


def test_model(model, test_data):
    """Test the model on the test data.

    Args:
        model (RegressionModel): Trained model.
        test_data (pd.DataFrame): Test data.

    Returns:
        np.array: Anomaly scores.
    """
    scores = model.get_scores(test_data.values)
    return scores


def plot_anomaly(data, scores, percentile=0.99):
    """Plot the anomaly scores.

    Args:
        data (pd.Series): Data.
        scores (np.array): Anomaly scores.
        percentile (float, optional): Percentile to use for threshold. Defaults to 0.99.

    Returns:
        matplotlib.figure.Figure: Figure.
    """
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


def train_regression(training_data: pd.Series, window: int = 10):
    """Train the regression model.

    Args:
        training_data (pd.Series): Training data.
        window (int, optional): Window size for the model. Defaults to 10.

    Returns:
        RegressionModel: Trained model.
    """
    model = RegressionModel(window=window)
    model.fit(training_data.values)
    return model


def prepare_dataset(input_data: pd.DataFrame,
                    descriptor_name: str = "AvgYieldPerCowFromDailyMilk",
                    farm_id: str = "24") -> pd.DataFrame:
    """Prepare the dataset for training.

    Args:
        df (pd.DataFrame): Raw data.
        descriptor_name (str, optional): Name of the descriptor to use. Defaults to "AvgYieldPerCowFromDailyMilk".
        farm_id (str, optional): Farm ID to use. Defaults to "24".
    Returns:
        pd.DataFrame: Processed dataset.
    """
    farm_data = input_data[input_data["farm_id"] == farm_id]
    farm_data.index = pd.to_datetime(farm_data.index)
    farm_data = farm_data.asfreq(freq='D')
    descriptor_data = farm_data[descriptor_name]
    descriptor_data.interpolate(inplace=True)
    return descriptor_data


if __name__ == '__main__':
    home = Path.home()
    data_path = home.joinpath("data")
    data_path.mkdir(exist_ok=True)
    print(f"Data path is: {data_path}")
    df = pd.read_parquet(data_path / "Production.parquet")

    descriptor_name = "AvgYieldPerCowFromDailyMilk"
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    dataset = prepare_dataset(input_data=df, descriptor_name=descriptor_name)

    TRAIN_SIZE = int(len(dataset) * 0.8)
    train, test = dataset[:TRAIN_SIZE], dataset[TRAIN_SIZE:]

    trained_model = train_regression(training_data=train)
    test_scores = test_model(model=trained_model, test_data=test)
    fig = plot_anomaly(data=test, scores=test_scores)

    fig.savefig(data_path / f"{descriptor_name}_{today}_test.png")

    with open(data_path / f"{descriptor_name}_{today}.pkl", "wb") as f:
        pickle.dump(trained_model, f)
