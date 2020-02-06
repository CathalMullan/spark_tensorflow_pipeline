"""
Parse environment into config.
"""
import os
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

    # Kubernetes
    cluster_ip: str = str(getenv("CLUSTER_IP"))

    # Google Cloud
    gcp_credentials: Path = Path(os.path.expanduser(str(getenv("GCP_CREDENTIALS")).strip()))
    gcp_project_id: str = str(getenv("GCP_PROJECT_ID"))


CONFIG = Config()
