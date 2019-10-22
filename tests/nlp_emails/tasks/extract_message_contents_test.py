"""
Test - extract_message_contents.py.
"""
from email.message import EmailMessage
from typing import Optional

import pytest

from nlp_emails.helpers.directories import TESTS_EMAIL_DIR
from nlp_emails.tasks.extract_message_contents import MessageContents, extract_message_contents
from nlp_emails.tasks.parse_email_messages import parse_email_messages

VALID_CONTENTS = MessageContents(
    message_id="hello@world.com",
    date=None,
    from_address="valid@email_1.com",
    to_address_list=["valid@email_2.com"],
    cc_address_list=None,
    bcc_address_list=None,
    subject="",
    body="Here is a valid body",
)

INVALID_CONTENTS_NO_FROM = MessageContents(
    message_id="hello@world.com",
    date=None,
    from_address="",
    to_address_list=["valid@email_2.com"],
    cc_address_list=None,
    bcc_address_list=["valid@email_2.com"],
    subject="",
    body="Here is a valid body",
)

INVALID_CONTENTS_NO_TO = MessageContents(
    message_id="hello@world.com",
    date=None,
    from_address="valid@email_1.com",
    to_address_list=[""],
    cc_address_list=["valid@email_2.com"],
    bcc_address_list=None,
    subject="",
    body="Here is a valid body",
)

INVALID_CONTENTS_NO_BODY = MessageContents(
    message_id="hello@world.com",
    date=None,
    from_address="valid@email_1.com",
    to_address_list=["valid@email_2.com"],
    cc_address_list=None,
    bcc_address_list=None,
    subject="",
    body="",
)

INVALID_CONTENTS_NO_MESSAGE_ID = MessageContents(
    message_id="",
    date=None,
    from_address="valid@email_1.com",
    to_address_list=["valid@email_2.com"],
    cc_address_list=None,
    bcc_address_list=None,
    subject="",
    body="Here is a valid body",
)

ALL_CONTENTS = [
    VALID_CONTENTS,
    INVALID_CONTENTS_NO_FROM,
    INVALID_CONTENTS_NO_TO,
    INVALID_CONTENTS_NO_BODY,
    INVALID_CONTENTS_NO_MESSAGE_ID,
]


@pytest.mark.parametrize("message", parse_email_messages(folder_path=TESTS_EMAIL_DIR))
def test_parse_message_to_dict(message: EmailMessage) -> None:
    """
    Test if email messages with a number of faults can be parsed to message contents.

    :param message: a parsed EmailMessage
    :return: None
    """
    message_contents: Optional[MessageContents] = extract_message_contents(message=message)

    if message_contents:
        assert isinstance(message_contents, MessageContents)
    else:
        assert message_contents is None


@pytest.mark.parametrize(
    "message_contents, is_valid",
    [
        (VALID_CONTENTS, True),
        (INVALID_CONTENTS_NO_FROM, False),
        (INVALID_CONTENTS_NO_TO, False),
        (INVALID_CONTENTS_NO_BODY, False),
        (INVALID_CONTENTS_NO_MESSAGE_ID, False),
    ],
)
def test_message_contents_validate(message_contents: MessageContents, is_valid: bool) -> None:
    """
    Test our validation around message contents. Specify the lack of/empty from, to, message-id headers and the body.

    :param message_contents: an instance of MessageContents
    :param is_valid: whether the contents should pass the validation
    :return: None
    """
    result: bool = message_contents.validate()

    if is_valid:
        assert result is True
    else:
        assert result is False


@pytest.mark.parametrize("message_contents", ALL_CONTENTS)
def test_message_contents_to_address_str(message_contents: MessageContents) -> None:
    """
    Ensure MessageContents attribute functions correctly.

    :param message_contents: an instance of MessageContents
    :return: None
    """
    to_address_str: Optional[str] = message_contents.to_address_str

    if to_address_str:
        assert isinstance(to_address_str, str)
    else:
        assert to_address_str is None


@pytest.mark.parametrize("message_contents", ALL_CONTENTS)
def test_message_contents_cc_address_str(message_contents: MessageContents) -> None:
    """
    Ensure MessageContents attribute functions correctly.

    :param message_contents: an instance of MessageContents
    :return: None
    """
    cc_address_str: Optional[str] = message_contents.cc_address_str

    if cc_address_str:
        assert isinstance(cc_address_str, str)
    else:
        assert cc_address_str is None


@pytest.mark.parametrize("message_contents", ALL_CONTENTS)
def test_message_contents_bcc_address_str(message_contents: MessageContents) -> None:
    """
    Ensure MessageContents attribute functions correctly.

    :param message_contents: an instance of MessageContents
    :return: None
    """
    bcc_address_str: Optional[str] = message_contents.bcc_address_str

    if bcc_address_str:
        assert isinstance(bcc_address_str, str)
    else:
        assert bcc_address_str is None


@pytest.mark.parametrize("message_contents", ALL_CONTENTS)
def test_message_contents_as_str(message_contents: MessageContents) -> None:
    """
    Ensure MessageContents attribute functions correctly.

    :param message_contents: an instance of MessageContents
    :return: None
    """
    message_contents_str: str = message_contents.as_str()

    if message_contents_str:
        assert isinstance(message_contents_str, str)
    else:
        assert message_contents_str is None


@pytest.mark.parametrize("message_contents", ALL_CONTENTS)
def test_message_contents_as_dict(message_contents: MessageContents) -> None:
    """
    Ensure MessageContents attribute functions correctly.

    :param message_contents: an instance of MessageContents
    :return: None
    """
    message_contents_dict = message_contents.as_dict()

    if message_contents_dict:
        assert isinstance(message_contents_dict, dict)
    else:
        assert message_contents_dict is None
