"""Module for anomaly detection models."""
from pathlib import Path
from typing import List, Optional

import pandas as pd
from darts import TimeSeries
from darts.ad import DifferenceScorer, ForecastingAnomalyModel, QuantileDetector
from darts.models import BlockRNNModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel


class VulcanAnomalyModel:
    """Class for anomaly detection models."""

    def __init__(self, input_chunk_length: int = 30, output_chunk_length: int = 1):
        """Initialize the model."""
        self._input_chunk_length = input_chunk_length
        self._output_chunk_length = output_chunk_length
        self._model: BlockRNNModel = BlockRNNModel(
            input_chunk_length=self._input_chunk_length,
            output_chunk_length=self._output_chunk_length,
            model="LSTM",
        )

    def load_model_from_path(self, model_path: Path):
        """
        Load the model from the given path.

        Args:
            model_path (Path): Path to the model file.
        Returns:
            BlockRNNModel: Trained model.
        """
        self._model = self._model.load(model_path.as_posix())
        return self._model

    def train_model(self, training_data: List[TimeSeries], epochs: int = 30):
        """
        Train the BlockRNN model.

        Args:
            training_data (pd.Series): Training data.
            epochs (int, optional): Number of epochs to train for. Defaults to 30.
        Returns:
            TorchForecastingModel: Trained model.
        """
        self._model.fit(training_data, epochs=epochs)
        return self._model

    def get_scores(self, ts_data: List[TimeSeries]) -> TimeSeries:
        """
        Get the scores for the given data.

        Args:
            ts_data (List[TimeSeries]): Data to score.
        Returns:
            TimeSeries: Model scores.
        """
        ds = DifferenceScorer()
        fam = ForecastingAnomalyModel(self._model, ds)
        scores = fam.score(ts_data, start=1)

        return scores

    def compute_anomalies(self, scores):
        """
        Compute the anomalies from the model scores.

        Args:
            scores (TimeSeries): Model scores.
        Returns:
            pd.DataFrame: Anomalies.
        """
        detector = QuantileDetector(high_quantile=0.998)
        scores = abs(scores)
        anomaly = detector.detect(scores)
        anomaly_df = pd.DataFrame(
            anomaly.values(), columns=["anomaly"], index=anomaly.time_index
        )
        anomaly_idx = anomaly_df[anomaly_df.anomaly == 1.0].index
        return anomaly_idx
