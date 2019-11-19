"""
Test - config.py.
"""
from distributed_nlp_emails.helpers.config.get_config import get_config
from distributed_nlp_emails.helpers.globals.directories import CONFIG_DIR, TESTS_EMAIL_DIR


def test_get_config() -> None:
    """
    Verify config path validation.

    :return:
    """
    assert isinstance(get_config(), dict)
    assert get_config("fake_path") is None
    assert get_config(CONFIG_DIR) is None
    assert get_config(TESTS_EMAIL_DIR + "/attachment_emails/attachment_content_disposition.eml") is None
