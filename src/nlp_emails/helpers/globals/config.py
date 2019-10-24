"""
Access contents of config fil, with overriding capabilities with default values.
"""
from typing import Any, Dict, Optional

from nlp_emails.helpers.config.get_config import get_config

CONFIG: Optional[Dict[str, Any]] = get_config()  # type: ignore
