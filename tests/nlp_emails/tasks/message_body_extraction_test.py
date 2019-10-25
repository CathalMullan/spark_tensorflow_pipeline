"""
Test - message_body_extraction.py.
"""
from email.message import EmailMessage
from typing import Optional

import pytest

from nlp_emails.helpers.globals.directories import TESTS_EMAIL_DIR
from nlp_emails.tasks.message_body_extraction import get_message_body
from nlp_emails.tasks.parse_email_messages import read_messages_from_directory


@pytest.mark.parametrize("message", read_messages_from_directory(TESTS_EMAIL_DIR))
def test_get_message_body(message: EmailMessage) -> None:
    """
    Ensure we can extract the body of an array of email messages.

    :param message: a parsed EmailMessage
    :return: None
    """
    message_body: Optional[str] = get_message_body(message=message)

    if message_body:
        assert isinstance(message_body, str)
    else:
        assert message_body is None
