"""Dataset processor class."""
from pathlib import Path
from typing import List, Union

import pandas as pd
from darts import TimeSeries


class DatasetProcessor:
    """Dataset processor class."""

    def __init__(self, dataset_path: Path):
        """Initialize the dataset processor."""
        self._dataset_path = dataset_path
        self._dataset_path.mkdir(parents=True, exist_ok=True)

    def process_descriptor(self, descriptor_name: str, descriptor_category: str):
        """Process the dataset for the given descriptor name and category."""
        raw_df = self._load_raw_dataset(data_category=descriptor_category)
        timeseries_data = self._prepare_dataset(
            descriptor_name=descriptor_name, dataset_df=raw_df
        )
        return timeseries_data

    def _load_raw_dataset(self, data_category: str):
        """Load the raw dataset from the given path."""
        dataset_df = pd.read_parquet(self._dataset_path / f"{data_category}.parquet")
        return dataset_df

    def _prepare_dataset(
        self,
        dataset_df: pd.DataFrame,
        descriptor_name: str = "AvgYieldPerCowFromDailyMilk",
    ) -> List[TimeSeries]:
        """Prepare the dataset for training.

        Args:
            descriptor_name (str, optional): Name of the descriptor to use. Defaults to "AvgYieldPerCowFromDailyMilk".
        Returns:
            pd.DataFrame: Processed dataset.
        """

        dataset_df.index = pd.to_datetime(dataset_df.index)
        dataset_df.index = dataset_df.index.tz_localize(None)
        total_data_as_time_series = [
            self._create_timeseries(
                data_df=dataset_df,
                farm_id=farm_id,
                descriptor_name=descriptor_name,
            )
            for farm_id in dataset_df.farm_id.unique()
        ]
        return total_data_as_time_series

    @staticmethod
    def _create_timeseries(
        data_df: pd.DataFrame, farm_id: Union[str, int], descriptor_name: str
    ):
        """Create a timeseries from the data."""
        df_farm = data_df[data_df["farm_id"] == farm_id]
        df_farm = df_farm.asfreq(freq="D")
        df_desc = df_farm[descriptor_name]
        df_desc.interpolate(inplace=True)
        return TimeSeries.from_series(df_desc)
