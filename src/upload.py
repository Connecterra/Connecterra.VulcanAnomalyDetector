"""Upload a model/experiment to Azure Blob."""
from pathlib import Path

from src.azure_utils.blob import upload_model_to_blob

if __name__ == '__main__':
    data_path = Path.home() / "data"
    model = data_path / "AvgYieldPerCowFromDailyMilk_2023-02-15.pkl"
    upload_model_to_blob(model_filepath=model)
