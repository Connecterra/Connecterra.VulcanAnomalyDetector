"""Upload a model/experiment to Azure Blob."""
from pathlib import Path

from src.utils.blob import AzureContainer

if __name__ == "__main__":
    data_path = Path.home() / "data"
    model = data_path / "AvgYieldPerCowFromDailyMilk_2023-03-07_blockrnn.pt"
    model_ckpt = (
        data_path / "AvgYieldPerCowFromDailyMilk_2023-03-07_blockrnn.pt.ckpt"
    )
    qd_path = (
        data_path / "AvgYieldPerCowFromDailyMilk_2023-03-07_quantiledetector.pkl"
    )
    container = AzureContainer()
    container.upload_model_to_blob(model_filepath=model)
    container.upload_model_to_blob(model_filepath=model_ckpt)
    container.upload_model_to_blob(model_filepath=qd_path)
