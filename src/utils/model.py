from pathlib import Path
from typing import List

from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel


class AnomalyModel:
    """Class for anomaly detection models."""

    def __init__(self, input_chunk_length: int = 30, output_chunk_length: int = 1):
        """Initialize the model."""
        self._input_chunk_length = input_chunk_length
        self._output_chunk_length = output_chunk_length

    def load_model(self, model_path: Path) -> TorchForecastingModel:
        """Load the model from the given path.

        Args:
            model_path (Path): Path to the model file.
        Returns:
            BlockRNNModel: Trained model.
        """
        model = BlockRNNModel(
            input_chunk_length=self._input_chunk_length,
            output_chunk_length=self._output_chunk_length,
            model="LSTM",
        )
        return model.load(model_path.as_posix())

    def train_model(
        self, training_data: List[TimeSeries], input_size: int = 30, epochs: int = 30
    ):
        """Train the BlockRNN model.

        Args:
            training_data (pd.Series): Training data.
            input_size (int, optional): Window size for the model. Defaults to 10.
            epochs (int, optional): Number of epochs to train for. Defaults to 30.
        Returns:
            TorchForecastingModel: Trained model.
        """
        output_chunk_length = 1
        model = BlockRNNModel(
            input_chunk_length=input_size,
            output_chunk_length=output_chunk_length,
            model="LSTM",
        )
        model.fit(training_data, epochs=epochs)

        return model
