from pydantic import BaseSettings, Field


class Config(BaseSettings):

    data_folder: str = Field(env="DATA_FOLDER")
    project: str = Field(env="PROJECT")
    logging_level: int = Field(env="LOGGING_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
