"""
Test - message_body_extraction.py.
"""
from email.message import EmailMessage
from typing import Optional

import pytest

from nlp_emails.helpers.directories import TESTS_EMAIL_DIR
from nlp_emails.tasks.message_body_extraction import get_message_body
from nlp_emails.tasks.parse_email_messages import parse_email_messages


@pytest.mark.parametrize("message", parse_email_messages(folder_path=TESTS_EMAIL_DIR))
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
