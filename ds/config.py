#!/usr/bin/env python3
"""Config class"""

from pydantic import BaseSettings, Field


class Config(BaseSettings):  # pylint: disable=too-few-public-methods
    """Config class"""

    data_folder: str = Field(env="DATA_FOLDER")
    project: str = Field(env="PROJECT")
    logging_level: int = Field(env="LOGGING_LEVEL")

    class Config:  # pylint: disable=too-few-public-methods
        """Configuration of Base Settings"""

        env_file = ".env"
        env_file_encoding = "utf-8"
