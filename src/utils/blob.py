import os
from pathlib import Path
from typing import Optional

from azure.storage.blob import BlobServiceClient


class AzureContainer:
    """
    Upload a model/experiment to Azure Blob.
    """

    def __init__(self, container_name: str = "store"):
        acc_name: Optional[str] = os.getenv("MODEL_REGISTRY_ACC", "modelregistry")
        acc_key: Optional[str] = os.getenv("MODEL_REGISTRY_KEY")

        self.service_client = BlobServiceClient.from_connection_string(
            conn_str=f"DefaultEndpointsProtocol=https;"
                     f"AccountName={acc_name};"
                     f"AccountKey={acc_key};"
                     f"EndpointSuffix=core.windows.net",
        )
        self.container_client = self.service_client.get_container_client(container_name)

    def upload_model_to_blob(
            self, model_filepath: Path,
            service_name: str = "vulcan"
    ):
        """Upload a model/experiment to Azure Blob."""

        model_name = model_filepath.name
        descriptor_name = model_name.split("_")[0]
        farm_id = model_name.split("_")[1]
        model_version = "_".join(model_name.split("_")[2:])
        blob_client = self.container_client.get_blob_client(
            blob=f"{service_name}/{descriptor_name}/{farm_id}/{model_version}"
        )

        with open(model_filepath, "rb") as data:
            results = blob_client.upload_blob(data, overwrite=True)

        return results

    def get_model_from_blob(self,
                            descriptor_name: str,
                            farm_id : str,
                            model_version: str,
                            service_name: str = "vulcan"
                            ):
        """Download a model/experiment from Azure blob."""
        blob_client = self.container_client.get_blob_client(
            blob=f"{service_name}/{descriptor_name}/{farm_id}/{model_version}"
        )
        blob_data = blob_client.download_blob()
        return blob_data
