"""Evaluate the anomaly detection model."""

import datetime
import logging
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
from ctra_charts.api.compatibility import (
    get_descriptors_compat,
)
from oats.threshold import QuantileThreshold

from src.utils.auth import Authenticator
from src.utils.blob import AzureContainer
from src.utils.charts_api import ChartsClient
from src.utils.dataset import DatasetProcessor
from src.utils.vulcan import VulcanAnomalyModel
from src.utils.mongodb import ProdMongo

logger = logging.getLogger("vulcan")

logging.getLogger("pytorch_lightning.utilities").propagate = False
logging.getLogger("pytorch_lightning.accelerators").propagate = False


class Evaluator:
    """Evaluator class."""

    def __init__(
        self, farm_id, descriptor_name: str, model_name: str, model_version: str
    ):
        """Evaluator class."""
        self.descriptor_name = descriptor_name
        self.farm_id = farm_id
        self.model_name = model_name
        self.model_version = model_version
        self._model: Optional[VulcanAnomalyModel] = None

        authenticator = Authenticator()
        auth = authenticator.authenticate()
        self.charts_client = ChartsClient(token=auth.token)
        self.model_descriptor = self.find_model_descriptor()

    def find_model_descriptor(self):
        """Get the model's trained descriptor from the ChartsAPI."""
        logger.info("Searching model descriptor")
        descriptors = get_descriptors_compat.sync(client=self.charts_client)
        model_descriptor = None
        for descriptor in descriptors:
            if descriptor.data_type_name == self.descriptor_name:
                model_descriptor = descriptor
                break
        if model_descriptor is None:
            raise ValueError(f"Descriptor {self.descriptor_name} not found.")
        logger.info(f"Found model descriptor id: {model_descriptor.id}")
        return model_descriptor

    def download_model(self, save_dir: Path):
        """Download the model from the Azure Blob."""
        logger.info("Downloading model from the Azure Blob.")
        container = AzureContainer()
        model_folder = container.get_model_from_blob(
            descriptor_name=self.descriptor_name,
            model_version=self.model_version,
            service_name=self.model_name,
            save_folder=save_dir / self.model_version,
        )
        # Return the file that ends with .pt in the contents of model_folder
        model_filepath = list(model_folder.glob("*.pt"))[0]

        return model_filepath

    def load_model(self, save_dir: Path):
        """Download & Load the model from Azure Blob."""
        # Save the model to the given directory
        model_filepath = self.download_model(save_dir)

        # Load the model from the downloaded file
        self._model = VulcanAnomalyModel()
        self._model.load_model_from_path(model_filepath)

        logger.info("Downloaded model file from Azure Blob successfully.")

    def evaluate_model(self, model_folder: Path):
        """Evaluate the model on the last 90 days of data."""
        if self._model is None:
            self.load_model(save_dir=model_folder)

        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.datetime.now() - datetime.timedelta(days=90)
        ).strftime("%Y-%m-%d")

        self.fetch_charts_data(
            start_date, end_date, self.model_descriptor.data_category, self.farm_id
        )

        charts_data_folder = Path(f"charts_{start_date}_{end_date}")
        data = DatasetProcessor(charts_data_folder).process_descriptor(
            descriptor_name=self.model_descriptor.data_type_name,
            descriptor_category=self.model_descriptor.data_category,
        )
        data = data[0]
        results = self._model.get_scores(data)
        anomalies = self._model.compute_anomalies(results)

        self.push_anomalies(data, anomalies)

    def push_anomalies(self, data: pd.Series, anomalies: pd.DataFrame):
        """Push the found anomalies to MongoDB."""
        ProdMongo().push_anomalies(
            data=data,
            anomalies=anomalies,
            farm_id=self.farm_id,
            descriptor_name=self.descriptor_name,
            model_name=self.model_name,
            model_version=self.model_version,
            data_category=self.model_descriptor.data_category,
            descriptor_id=self.model_descriptor.id,
        )

    def extract_anomalies(self, results, percentile=0.99):
        """Extract the anomalies from the model results."""
        logger.info("Extracting anomalies.")
        threshold = QuantileThreshold()
        threshold_values = threshold.get_threshold(
            data=results, percentile=percentile
        )
        anomalies = results > threshold_values
        logger.info(f"Found {anomalies.sum()} anomalies.")
        return anomalies

    @staticmethod
    def fetch_charts_data(
        start_date: str,
        end_date: str,
        category: str,
        farm_id: str,
    ):
        """
        Fetch data from the ChartsAPI.

        Args:
            start_date (datetime.datetime): Start date of the data to fetch.
            end_date (datetime.datetime): End date of the data to fetch.
            category (str): Category of the data to fetch.
            farm_id (str): Farm ID of the data to fetch.
        """
        args = [
            "python",
            "-m",
            "kpi_dataset_creator",
            "run",
            "--start_date",
            start_date,
            "--end_date",
            end_date,
            "--categories",
            category,
            "--farm_ids",
            farm_id,
        ]
        logger.info(
            f"Fetching charts data with dataset creator using: {' '.join(args)}"
        )
        process = subprocess.run(args, capture_output=True)
        logger.info(process.stdout)
        logger.info(process.stderr)


if __name__ == "__main__":
    model_folder = Path("/home/efehan/data")
    evaluator = Evaluator(
        farm_id="153",
        descriptor_name="AvgYieldPerCowFromDailyMilk",
        model_name="vulcan",
        model_version="2023-03-07",
    )
    evaluator.evaluate_model(model_folder=model_folder)
