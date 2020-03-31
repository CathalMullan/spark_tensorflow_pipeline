"""
Parse environment into a config object.
"""
from dataclasses import dataclass
from os import getenv
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from spark_tensorflow_pipeline.helpers.globals.directories import PROJECT_DIR

load_dotenv(dotenv_path=f"{PROJECT_DIR}/.env")


def is_true(variable: Optional[str]) -> bool:
    """
    Validate string comparision to handle boolean environment variables.

    :param variable: environment variable
    :return: boolean if 'true'
    """
    return variable == "true"


@dataclass
class Config:
    """
    Config object.
    """

    # Generic
    is_dev: bool = is_true(getenv("IS_DEV"))

    # Google Cloud
    gcp_credentials: Path = Path(str(getenv("GCP_CREDENTIALS")))

    # Bucket
    bucket_parquet: str = str(getenv("BUCKET_PARQUET"))

    # Dictionary
    dictionary_path: str = PROJECT_DIR + "/words.txt"

    # Topic Modelling
    bucket_topic_model: str = str(getenv("BUCKET_TOPIC_MODEL"))
    bucket_saved_topic_model: str = str(getenv("BUCKET_SAVED_TOPIC_MODEL"))

    # Process Date
    process_date: Optional[str] = getenv("PROCESS_DATE")


CONFIG = Config()
