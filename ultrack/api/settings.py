from pathlib import Path
from typing import Union

from pydantic.v1 import BaseSettings

from ultrack.config import DataConfig


class Settings(BaseSettings):
    """Settings for the API.

    Attributes
    ----------
    api_results_path : Union[str, Path, None]
        Path to the API results folder. It stores any file/folder of the results from
        intermediate processing steps and final results. Defaults to "/tmp".
    ultrack_data_config : DataConfig
        Ultrack data configuration. It is used to store the configure the database
        connection and the working directory. Defaults to DataConfig().
    """

    api_results_path: Union[str, Path, None] = Path("/tmp")
    ultrack_data_config: DataConfig = DataConfig()


settings = Settings()
