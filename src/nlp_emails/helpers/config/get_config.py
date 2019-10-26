"""
Attempt to read config path, parse contents to a dictionary.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union

import toml

from nlp_emails.helpers.globals.directories import CONFIG_DIR


def get_config(config_path: Union[Path, str] = CONFIG_DIR + "/config.toml") -> Optional[Dict[str, Any]]:  # type: ignore
    """
    Attempt to read config path, parse contents to a dictionary.

    :param config_path: string or Path to config file
    :return: parsed dict of file contents
    """
    if not isinstance(config_path, Path):
        config_path = Path(config_path)

    if not config_path.exists() or not config_path.is_file():
        return None

    parsed_toml = toml.load(config_path)

    toml_dict = dict(parsed_toml)
    return toml_dict


CONFIG: Optional[Dict[str, Any]] = get_config()  # type: ignore
