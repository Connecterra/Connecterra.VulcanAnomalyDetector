"""Evaluate the anomaly detection model."""

import datetime
import io
import pickle
import subprocess
from pathlib import Path
from typing import List
import logging

import numpy as np
import pandas as pd
from ctra_charts.api.compatibility import (
    get_descriptors_compat,
)
from oats.threshold import QuantileThreshold

from src.utils.auth import Authenticator
from src.utils.blob import AzureContainer
from src.utils.charts_api import ChartsClient
from src.utils.mongodb import ProdMongo

logger = logging.getLogger("vulcan")


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
        self.model = None

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

    def load_model(self):
        """Load the model from Azure Blob."""
        logger.info("Downloading model file from Azure Blob.")
        container = AzureContainer()
        model_stream_data = container.get_model_from_blob(
            descriptor_name=self.descriptor_name,
            farm_id=self.farm_id,
            model_version=self.model_version,
            service_name=self.model_name,
        )
        model_bytes = io.BytesIO()
        model_stream_data.readinto(model_bytes)
        self.model = pickle.loads(model_bytes.getvalue())
        logger.info("Downloaded model file from Azure Blob successfully.")

    def evaluate_model(self):
        """Evaluate the model on the last 90 days of data."""
        if self.model is None:
            self.load_model()

        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.datetime.now() - datetime.timedelta(days=90)
        ).strftime("%Y-%m-%d")

        self.fetch_charts_data(
            start_date, end_date, self.model_descriptor.data_category, self.farm_id
        )

        charts_parquet_filepath = (
            Path(f"charts_{start_date}_{end_date}")
            / f"{self.model_descriptor.data_category}.parquet"
        )
        data = self.preprocess_data(charts_parquet_filepath)
        results = self.model.get_scores(data.values)
        anomalies = self.extract_anomalies(results)

        self.push_anomalies(data, anomalies)

    def push_anomalies(self, data: pd.Series, anomalies: np.ndarray):
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

    def preprocess_data(self, charts_parquet_filepath):
        """Preprocess the data for the model."""
        charts_data = pd.read_parquet(charts_parquet_filepath)
        charts_data = charts_data[charts_data["farm_id"] == self.farm_id]
        charts_series = charts_data[self.descriptor_name]
        charts_series.index = pd.to_datetime(charts_series.index)
        charts_series = charts_series.asfreq(freq="D")
        charts_series.interpolate(inplace=True)

        return charts_series

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
        subprocess.run(
            args,
            capture_output=True,
        )


if __name__ == "__main__":
    evaluator = Evaluator(
        farm_id="24",
        descriptor_name="AvgYieldPerCowFromDailyMilk",
        model_name="vulcan",
        model_version="2023-02-15.pkl",
    )
    evaluator.evaluate_model()
