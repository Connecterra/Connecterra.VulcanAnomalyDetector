"""Training script for the model."""
import datetime
import pickle
from pathlib import Path
from typing import Tuple, Sequence, Union, List
import logging

import pandas as pd
from darts import TimeSeries
from darts.ad import QuantileDetector, ForecastingAnomalyModel, DifferenceScorer
from darts.models import BlockRNNModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from src.utils.dataset import DatasetProcessor
from src.utils.vulcan import VulcanAnomalyModel

logging.getLogger("pytorch_lightning.utilities").propagate = False
logging.getLogger("pytorch_lightning.accelerators").propagate = False


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
    ) -> Tuple[TorchForecastingModel, QuantileDetector]:
        """Train the model on the training data for the given descriptor name."""
        logging.info(f"Training model for descriptor: {descriptor_name}")
        training_data = DatasetProcessor(dataset_path=data_path).process_descriptor(
            descriptor_name=descriptor_name, descriptor_category=descriptor_category
        )
        model = VulcanAnomalyModel(
            input_chunk_length=self._input_chunk_length,
            output_chunk_length=self._output_chunk_length,
        )
        trained_model = model.train_model(training_data=training_data, epochs=epochs)
        logging.info(f"Training complete for descriptor: {descriptor_name}")
        quantile_detector = self.train_quantile_detector(
            trained_model=trained_model, training_data=training_data
        )

        return trained_model, quantile_detector

    def train_quantile_detector(
        self,
        trained_model: Union[TorchForecastingModel, BlockRNNModel],
        training_data: Union[Sequence[TimeSeries], TimeSeries],
    ) -> QuantileDetector:
        """Train the quantile detector on the given model and training data."""
        logging.info("Training quantile detector")
        diff_scorer = DifferenceScorer()
        forecasting_anomaly_model = ForecastingAnomalyModel(
            trained_model, diff_scorer
        )
        scores = self.create_scores(forecasting_anomaly_model, training_data)
        quantile_detector = QuantileDetector(high_quantile=0.998)
        quantile_detector = quantile_detector.fit(scores)
        return quantile_detector

    def create_scores(
        self,
        forecasting_model: ForecastingAnomalyModel,
        training_data: Union[Sequence[TimeSeries], TimeSeries],
    ) -> TimeSeries:
        """Creates scores from list of time series and \
        combine them into a TimeSeries object"""
        scores = list()
        for series in training_data[:2]:
            scores.append(forecasting_model.score(series, start=1))
        scores_combined = scores[0]
        for series in scores[1:]:
            scores_combined = scores_combined.concatenate(
                series, ignore_time_axis=True
            )
        scores_combined = abs(scores_combined)
        return scores_combined


if __name__ == "__main__":
    home = Path.home()
    data_path = home.joinpath("data")
    trainer = VulcanTrainer()
    descriptor_name = "AvgYieldPerCowFromDailyMilk"
    descriptor_category = "Production"

    today = datetime.datetime.today().strftime("%Y-%m-%d")
    model_filepath = data_path / f"{descriptor_name}_{today}_blockrnn.pt"

    model, quantile_detector = trainer.train_descriptor(
        data_path=data_path,
        descriptor_category=descriptor_category,
        descriptor_name=descriptor_name,
        epochs=2,
    )
    model.save(model_filepath.as_posix())
    with open(
        data_path / f"{descriptor_name}_{today}_quantiledetector.pkl", "wb"
    ) as f:
        pickle.dump(quantile_detector, f)
