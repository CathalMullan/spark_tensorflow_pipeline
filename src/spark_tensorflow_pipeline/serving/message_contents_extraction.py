"""
Extract headers and body from email message.
"""
from email import message_from_string
from email.errors import MessageDefect
from email.message import EmailMessage
from email.policy import strict
from typing import Dict, Optional

from dataclasses import dataclass

from spark_tensorflow_pipeline.serving.message_body_extraction import get_message_body
from spark_tensorflow_pipeline.serving.message_header_extraction import get_message_raw_headers, get_message_subject


@dataclass
class MessageContent:
    """
    Select components and headers of a parsed EmailMessage.
    """

    subject: str = ""
    body: str = ""

    def validate(self) -> bool:
        """
        Verify if MessageContents instance is valid for further processing.

        :return: bool whether message_contents is valid
        """
        # Require a message body of reasonable length
        # if not self.body or not is_valid_length(text=self.body, minimum=200, maximum=5_000):
        #     print("Invalid body length.")
        #     return False

        return True


def read_message_from_string(message_str: str) -> Optional[EmailMessage]:
    """
    Parse a string to an EmailMessage.

    :param message_str: eml file as a string
    :return: parsed EmailMessage from string
    """
    try:
        # the policy 'strict' makes this return an EmailMessage class (Python 3.6+), rather than a Message class.
        # noinspection PyTypeChecker
        email_message: EmailMessage = message_from_string(message_str, policy=strict)  # type: ignore
    except MessageDefect as message_defects:
        print(f"Could not parse string to EmailMessage: {message_defects}")
        return None

    return email_message


def is_valid_length(text: str, minimum: int, maximum: int) -> bool:
    """
    Ensure text is between a min and max.

    :param text: text to measure length
    :param minimum: lower bound integer
    :param maximum: higher bound integer
    :return: bool if valid
    """
    text_len = len(text)
    if text_len > maximum or text_len < minimum:
        return False

    return True


def extract_message_contents(message: EmailMessage) -> Optional[MessageContent]:
    """
    Extract fields from a message to a dict of contents.

    :param message: a parsed EmailMessage
    :return: optional parsed message content
    """
    # Validate required headers are present.
    raw_headers: Optional[Dict[str, str]] = get_message_raw_headers(message=message)
    if not raw_headers:
        print("Invalid headers.")
        return None

    # Build message contents
    message_contents = MessageContent(
        subject=get_message_subject(subject_header_str=raw_headers.get("subject")),
        body=get_message_body(message=message),
    )

    # Validate final state of message.
    if not message_contents.validate():
        return None

    return message_contents


def eml_str_to_message_contents(eml_str: str) -> Optional[MessageContent]:
    """
    Process eml file as bytes and convert to message content dict.

    :param eml_str: str representation of an eml file
    :return: optional message content
    """
    email_message: Optional[EmailMessage] = read_message_from_string(eml_str.strip())
    if not isinstance(email_message, EmailMessage):
        return None

    message_contents: Optional[MessageContent] = extract_message_contents(email_message)
    if not isinstance(message_contents, MessageContent):
        print(f"Could not parse email to MessageContent")
        return None

    return message_contents
