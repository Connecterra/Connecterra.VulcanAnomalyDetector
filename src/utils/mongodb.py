import datetime
import logging
import os

import numpy as np
import pandas as pd
from pymongo import MongoClient

logger = logging.getLogger("vulcan")


class ProdMongo(MongoClient):
    """
    MongoDB client for production.
    """

    def __init__(self):
        host = os.getenv("MONGO_HOST")
        username = os.getenv("MONGO_USERNAME")
        password = os.getenv("MONGO_PASSWORD")
        super().__init__(
            host=host,
            username=username,
            password=password,
        )
        if host.startswith("cs.mongo.internal.connecterra.io"):
            self.db = self.get_database("IDA_MASTER_PROD")
        else:
            self.db = self.get_database("IDA_MASTER")

    def batch_push_documents(self, documents: list, collection_name: str):
        """
        Push a batch of documents to a collection.
        """
        logger.info(
            f"Pushing {len(documents)} documents to {collection_name} on {self.db.name} DB."
        )
        collection = self.db.get_collection(collection_name)
        collection.insert_many(documents)

    def push_anomalies(
        self,
        data: pd.Series,
        anomalies: np.ndarray,
        farm_id: str,
        descriptor_name: str,
        model_name: str,
        model_version: str,
        data_category: str,
        descriptor_id: str,
    ):
        """
        Push a batch of anomalies to the anomalies collection.
        """
        anomaly_dates = data.index[anomalies]
        anomaly_docs = []
        for anomaly_date in anomaly_dates:
            anomaly = {
                "farmId": farm_id,
                "descriptorName": descriptor_name,
                "model": model_name,
                "modelVersion": model_version,
                "dataCategory": data_category,
                "descriptorId": descriptor_id,
                "eventDate": anomaly_date,
                "anomalyScore": data[anomaly_date],
                "modifiedDate": None,
                "createdDate": datetime.datetime.now(),
            }
            anomaly_docs.append(anomaly)
        self.batch_push_documents(anomaly_docs, "DS_DataAnomalies")
