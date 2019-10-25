"""
Extract headers and body from email message.
"""
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Union

from nlp_emails.tasks.message_body_extraction import get_message_body
from nlp_emails.tasks.message_header_extraction import (
    get_message_address,
    get_message_address_list,
    get_message_date,
    get_message_message_id,
    get_message_raw_headers,
    get_message_subject,
)
from nlp_emails.tasks.parse_email_messages import read_message_from_file


@dataclass
class MessageContent:
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

    @property
    def to_address_str(self) -> Optional[str]:
        """
        Concatenate to_address_list into a comma separated list.

        :return: string representation of to_address_list
        """
        if not self.to_address_list or all(address == "" for address in self.to_address_list):
            return None

        return ", ".join(self.to_address_list).strip()

    @property
    def cc_address_str(self) -> Optional[str]:
        """
        Concatenate cc_address_list into a comma separated list.

        :return: string representation of cc_address_list
        """
        if not self.cc_address_list or all(address == "" for address in self.cc_address_list):
            return None

        return ", ".join(self.cc_address_list).strip()

    @property
    def bcc_address_str(self) -> Optional[str]:
        """
        Concatenate bcc_address_list into a comma separated list.

        :return: string representation of bcc_address_list
        """
        if not self.bcc_address_list or all(address == "" for address in self.bcc_address_list):
            return None

        return ", ".join(self.bcc_address_list).strip()

    def validate(self) -> bool:
        """
        Verify if MessageContents instance is valid for further processing.

        :return: bool as to whether message_contents is valid
        """
        # Require a valid message-id (generated or otherwise)
        if not self.message_id:
            return False

        # Require a valid from address
        if not self.from_address:
            return False

        # Require at least one valid to address
        if not self.to_address_list or all(address == "" for address in self.to_address_list):
            return False

        # Require a message body
        if not self.body:
            return False

        return True

    def as_str(self) -> Optional[str]:
        """
        Convert MessageContents instance into an eml like string.

        :return: eml file of message
        """
        if not self.validate():
            return None

        return textwrap.dedent(
            f"""
            Message-Id: {self.message_id}
            Date: {self.date}
            From: {self.from_address}
            To: {self.to_address_str}
            Cc: {self.cc_address_str}
            Bcc: {self.bcc_address_str}
            Subject: {self.subject}

            {self.body}
            """
        ).strip()

    def as_dict(self) -> Optional[Dict[str, Union[Optional[str], Optional[datetime]]]]:
        """
        Convert MessageContents instance into a dict.

        :return: dict of contents
        """
        if not self.validate():
            return None

        return {
            "message_id": self.message_id,
            "date": self.date,
            "from_address": self.from_address,
            "to_address": self.to_address_str,
            "cc_address": self.cc_address_str,
            "bcc_address_list": self.bcc_address_str,
            "subject": self.subject,
            "body": self.body,
        }


def extract_message_contents(message: EmailMessage) -> Optional[MessageContent]:
    """
    Extract fields from a message to a dict of contents.

    :param message: a parsed EmailMessage
    :return: optional parsed message content
    """
    message_contents = MessageContent()

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

    if not message_contents.validate():
        return None

    return message_contents


def eml_path_to_message_contents(eml_path: Path) -> Optional[MessageContent]:
    """
    Read eml file and convert to message content.

    :param eml_path: path to an eml file
    :return: optional message content
    """
    email_message: Optional[EmailMessage] = read_message_from_file(eml_path)
    if not isinstance(email_message, EmailMessage):
        return None

    message_contents: Optional[MessageContent] = extract_message_contents(email_message)
    if not isinstance(message_contents, MessageContent):
        return None

    return message_contents
