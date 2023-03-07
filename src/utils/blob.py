import os
from pathlib import Path
from typing import Optional

from azure.storage.blob import BlobServiceClient


class AzureContainer:
    """
    Upload a model/experiment to Azure Blob.
    """

    def __init__(self, container_name: str = "store"):
        self._acc_name: Optional[str] = os.getenv(
            "MODEL_REGISTRY_ACC", "modelregistry"
        )
        self._acc_key: Optional[str] = os.getenv("MODEL_REGISTRY_KEY")

        self.service_client = BlobServiceClient.from_connection_string(
            conn_str=f"DefaultEndpointsProtocol=https;"
            f"AccountName={self._acc_name};"
            f"AccountKey={self._acc_key};"
            f"EndpointSuffix=core.windows.net",
        )
        self.container_client = self.service_client.get_container_client(
            container_name
        )

    def upload_model_to_blob(
        self, model_filepath: Path, service_name: str = "vulcan"
    ):
        """Upload a model/experiment to Azure Blob."""

        model_fullname = model_filepath.name
        model_path = "/".join(model_fullname.split("_"))
        blob_client = self.container_client.get_blob_client(
            blob=f"{service_name}/{model_path}"
        )

        with open(model_filepath, "rb") as data:
            results = blob_client.upload_blob(data, overwrite=True)

        return results

    def get_model_from_blob(
        self,
        descriptor_name: str,
        model_version: str,
        save_folder: Path,
        service_name: str = "vulcan",
    ):
        """Download a model/experiment from Azure blob."""

        model_blob_folder = f"{service_name}/{descriptor_name}/{model_version}"
        model_blobs = self.container_client.list_blobs(model_blob_folder)

        for blob in model_blobs:
            file_contents = (
                self.container_client.get_blob_client(blob=blob)
                .download_blob()
                .readall()
            )
            self.save_blob(
                save_folder=save_folder,
                file_name=blob.name.split("/")[-1],
                file_content=file_contents,
            )

        return save_folder

    def save_blob(self, save_folder, file_name, file_content):
        # for nested blobs, create local path as well!
        os.makedirs(save_folder, exist_ok=True)

        with open(os.path.join(save_folder, file_name), "wb") as file:
            file.write(file_content)
