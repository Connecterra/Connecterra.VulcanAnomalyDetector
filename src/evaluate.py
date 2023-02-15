import pickle
from pathlib import Path

from src.azure_utils.blob import AzureContainer


if __name__ == '__main__':
    container = AzureContainer()
    model_pkl = container.get_model_from_blob(
        local_path=Path("model.pkl"),
        model_category="AvgYieldPerCowFromDailyMilk",
        model_version="2023-02-15.pkl",
    )

    model = pickle.loads(model_pkl)
