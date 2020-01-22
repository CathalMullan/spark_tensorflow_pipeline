"""
Parse environment into config.
"""
import os
from dataclasses import dataclass
from os import getenv
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from distributed_nlp_emails.helpers.globals.directories import PROJECT_DIR

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
    is_dev: bool = is_true("IS_DEV")

    # Message Extraction
    do_content_tagging: bool = is_true(getenv("DO_CONTENT_TAGGING"))
    do_faker_replacement: bool = is_true(getenv("DO_FAKER_REPLACEMENT"))
    do_address_hashing: bool = is_true(getenv("DO_ADDRESS_HASHING"))

    # Kubernetes
    cluster_ip: str = str(getenv("CLUSTER_IP"))

    # Google Cloud
    gcp_credentials: Path = Path(os.path.expanduser(str(getenv("GCP_CREDENTIALS")).strip()))
    gcp_project_id: str = str(getenv("GCP_PROJECT_ID"))

    # Spark (Batch)
    spark_gcp_credentials: str = str(getenv("SPARK_GCP_CREDENTIALS"))
    spark_gcp_parquet: str = str(getenv("SPARK_GCP_PARQUET"))


CONFIG = Config()
