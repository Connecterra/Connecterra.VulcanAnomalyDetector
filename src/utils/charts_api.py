"""ChartsAPI Helper module."""
import os

from ctra_charts import AuthenticatedClient


class ChartsClient(AuthenticatedClient):
    """ChartsAPI Client."""

    BASE_URL = os.getenv(
        "CHARTSAPI_URL", r"https://ctra-charts-staging.azurewebsites.net"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, base_url=self.BASE_URL, timeout=20, **kwargs)
