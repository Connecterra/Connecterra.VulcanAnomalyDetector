"""Training script for the model."""
from pathlib import Path

import pandas as pd
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from src.utils.dataset import DatasetProcessor
from src.utils.model import AnomalyModel


class VulcanTrainer:
    """Trainer class."""

    def __init__(self, input_chunk_length: int = 30, output_chunk_length: int = 1):
        """Initialize the trainer."""
        self._input_chunk_length = input_chunk_length
        self._output_chunk_length = output_chunk_length
        self._dataset_df = pd.DataFrame()

    def train_descriptor(
        self,
        data_path: Path,
        descriptor_category: str,
        descriptor_name: str,
        epochs: int = 30,
    ) -> TorchForecastingModel:
        """Train the model on the training data for the given descriptor name."""
        training_data = DatasetProcessor(dataset_path=data_path).process_descriptor(
            descriptor_name=descriptor_name, descriptor_category=descriptor_category
        )
        model = AnomalyModel(
            input_chunk_length=self._input_chunk_length,
            output_chunk_length=self._output_chunk_length,
        )
        trained_model = model.train_model(training_data=training_data, epochs=epochs)
        return trained_model


if __name__ == "__main__":
    home = Path.home()
    data_path = home.joinpath("data")
    trainer = VulcanTrainer()
    model = trainer.train_descriptor(
        data_path=data_path,
        descriptor_category="Production",
        descriptor_name="AvgYieldPerCowFromDailyMilk",
        epochs=5,
    )

    model.save((data_path / f"blockrnn_model.pt").as_posix())
