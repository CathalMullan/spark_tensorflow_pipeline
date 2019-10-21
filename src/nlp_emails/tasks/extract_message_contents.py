"""
Extract headers and body from email message.
"""
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple

from nlp_emails.tasks.message_body_extraction import get_message_body
from nlp_emails.tasks.message_header_extraction import (
    get_message_address,
    get_message_address_list,
    get_message_date,
    get_message_message_id,
    get_message_subject,
)


@dataclass
class MessageContents:
    """
    Select components and headers of a parsed EmailMessage.
    """

    message_id: str = ""
    date: Optional[datetime] = None
    from_address: str = ""
    to_address_list: List[str] = field(default_factory=list)
    cc_address_list: Optional[List[str]] = None
    bcc_address_list: Optional[List[str]] = None
    subject: str = ""
    body: str = ""


def valid_message_contents(message_contents: MessageContents) -> bool:
    """
    Verify if a MessageContents instance is valid for further processing.

    TODO: There's probably a nicer way to integrate this into the dataclass.

    :param message_contents: an instance of MessageContents
    :return: bool as to whether message_contents is valid
    """
    # Require a valid message-id (generated or otherwise)
    if not message_contents.message_id:
        return False

    # Require a valid from address
    if not message_contents.from_address:
        return False

    # Require at least one valid to address
    if not message_contents.to_address_list or all(address == "" for address in message_contents.to_address_list):
        return False

    # Require a message body
    if not message_contents.body:
        return False

    return True


def get_message_raw_headers(message: EmailMessage) -> Optional[Dict[str, str]]:
    """
    Extract the header keys and values from a email message.

    Handle parsing errors that are common with individual headers.

    :param message: a parsed EmailMessage
    :return: optional dictionary of header keys and values
    """
    raw_headers: Dict[str, str] = {}

    try:
        header_list: List[Tuple[str, str]] = message.items()
    except TypeError:
        print(f"Header Error: Cannot parse header list")
        return None

    for header_key, header_value in header_list:
        if header_key.lower() in ["message-id", "date", "from", "to", "cc", "bcc", "subject"]:
            raw_headers[header_key] = str(header_value)

    return dict(raw_headers)


def extract_message_contents(message: EmailMessage) -> Optional[MessageContents]:
    """
    Extract fields from a message to a dict of contents.

    :param message: a parsed EmailMessage
    :return: optional parsed fields in a dict
    """
    message_contents = MessageContents()

    raw_headers: Optional[Dict[str, str]] = get_message_raw_headers(message=message)
    if not raw_headers:
        return None

    message_contents.message_id = get_message_message_id(message_id_str=raw_headers.get("message-id"))
    message_contents.date = get_message_date(date_header_str=raw_headers.get("date"))

    message_contents.from_address = get_message_address(header_str=raw_headers.get("from"))
    message_contents.to_address_list = get_message_address_list(header_str=raw_headers.get("to"))
    message_contents.cc_address_list = get_message_address_list(header_str=raw_headers.get("cc"))
    message_contents.bcc_address_list = get_message_address_list(header_str=raw_headers.get("bcc"))

    message_contents.subject = get_message_subject(subject_header_str=raw_headers.get("subject"))
    message_contents.body = get_message_body(message=message)

    return message_contents
