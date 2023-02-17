"""Upload a model/experiment to Azure Blob."""
from pathlib import Path

from src.utils.blob import AzureContainer

if __name__ == '__main__':
    data_path = Path.home() / "data"
    model = data_path / "AvgYieldPerCowFromDailyMilk_24_2023-02-15.pkl"
    container = AzureContainer()
    container.upload_model_to_blob(model_filepath=model)
