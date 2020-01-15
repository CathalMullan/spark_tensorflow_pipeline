"""
Parse environment into config.
"""
from dataclasses import dataclass
from os import getenv
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

    # Spark Configurations
    cluster_ip: str = str(getenv("CLUSTER_IP"))


CONFIG = Config()
