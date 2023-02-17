"""ChartsAPI Authentication Module."""
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Union, Optional

import dateutil.parser
import requests
from requests import RequestException

logger = logging.getLogger()


@dataclass
class AuthResponse:
    """ChartsAPI Authentication Response Model."""

    userId: str
    keyId: str
    username: str
    email: str
    token: str
    expirationDate: Union[str, datetime]
    expirationEpoch: int
    refreshToken: str
    refreshTokenExpirationDate: Union[str, datetime]
    refreshTokenExpirationEpoch: int

    def __post_init__(self):
        """Convert the date datatypes from string to datetime format."""
        self.expirationDate = dateutil.parser.parse(self.expirationDate)
        self.refreshTokenExpirationDate = dateutil.parser.parse(
            self.refreshTokenExpirationDate
        )


class Authenticator:
    """ChartsAPI Authenticator Class."""

    TOKEN_URL = r"https://acf-auth-staging.azurewebsites.net/token"
    TOKEN_REFRESH_URL = r"https://acf-auth-staging.azurewebsites.net/token-renewal"

    def __init__(
            self, username: Optional[str] = None, password: Optional[str] = None
    ):
        """Init Constructor for Authenticator class."""
        if username is None:
            username = os.getenv("CHARTSAPI_USERNAME")
        if password is None:
            password = os.getenv("CHARTSAPI_PASSWORD")
        self.username = username
        self.password = password

        self.auth: Optional[AuthResponse] = None

    def authenticate(self):
        """Send the authentication request to ChartsAPI auth server."""
        if self.is_authenticated():
            logger.info("Already authenticated.")
            return self.auth
        elif (
                self.auth is not None
                and self.auth.refreshTokenExpirationDate > datetime.now(tz=timezone.utc)
        ):
            logger.info("Refreshing authentication token.")
            return self.refresh_token()
        auth_params = {"username": self.username, "password": self.password}
        logger.info("Authenticating...")
        self.post_authentication(self.TOKEN_URL, auth_params)
        return self.auth

    def post_authentication(self, url: str, auth_params: dict):
        """Post the authentication request to the ChartsAPI auth server."""
        with requests.post(url, json=auth_params) as r:
            resp = r.json()
            if not r.status_code == 200:
                logger.error(
                    f"Authentication errored with status code {r.status_code}.",
                    extra={"details": r.text},
                )
                raise RequestException(resp)
        self.auth = AuthResponse(**resp)
        logger.info("Authentication successful.")

    def refresh_token(self):
        """Refresh the authentication token."""
        if self.auth is None:
            raise ValueError("No authentication token to refresh.")
        if self.auth.expirationDate > datetime.now(tz=timezone.utc):
            logger.info("Authentication token is still valid.")
            return self.auth
        elif self.auth.refreshTokenExpirationDate > datetime.now(timezone.utc):
            logger.info("Refresh token has expired.")
            return self.authenticate()
        refresh_params = {
            "refreshToken": self.auth.refreshToken,
            "expiredToken": self.auth.token,
        }
        logger.info("Refreshing authentication token...")
        self.post_authentication(self.TOKEN_REFRESH_URL, refresh_params)
        logger.info("Authentication token refreshed.")
        return self.auth

    def is_authenticated(self):
        """Check if authentication happened."""
        return self.auth is not None and self.auth.expirationDate > datetime.now(
            tz=timezone.utc
        )
