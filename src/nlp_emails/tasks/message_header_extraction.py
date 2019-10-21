"""
Functions to extract contents from message headers.
"""
from datetime import datetime
from email.utils import make_msgid, mktime_tz, parsedate_tz
from typing import List, Optional

from nlp_emails.helpers.address_validation import parse_address_str
from nlp_emails.helpers.regex import SUBJECT_PREFIX
from nlp_emails.helpers.text_validation import ensure_language_english


def get_message_address(header_str: str) -> Optional[str]:
    """
    Get a message header as a single valid address string.

    :param header_str: the message header as a string
    :return: optional valid address from the message header
    """
    if not header_str:
        print(f"Header Error - No address header found: {header_str}")
        return None

    parsed_address: Optional[str] = parse_address_str(potential_address=header_str)
    if not parsed_address:
        print(f"Header Error - Cannot parse address header: {header_str}")
        return None

    return parsed_address


def get_message_address_list(header_str: str) -> Optional[List[str]]:
    """
    Get a message header as a list of valid address strings.

    :param header_str: the message header as a string
    :return: optional list of valid addresses strings from the message header
    """
    if not header_str:
        print(f"Header Error - No address list header found: {header_str}")
        return None

    split_header_str: List[str] = header_str.split(", ")

    parsed_header_addresses: List[str] = []
    for potential_address in split_header_str:
        parsed_address: Optional[str] = parse_address_str(potential_address=potential_address)
        if parsed_address:
            parsed_header_addresses.append(parsed_address)

    if not parsed_header_addresses:
        print(f"Header Error - Cannot parse address list header: {header_str}")
        return None

    return parsed_header_addresses


def get_message_date(date_header_str: str) -> Optional[datetime]:
    """
    Get the message date header as a datetime.

    :param date_header_str: the message date header as a string
    :return: optional datetime from the date_header
    """
    if not date_header_str:
        print(f"Header Error - No date header found: {date_header_str}")
        return None

    date_header_tuple = parsedate_tz(date_header_str)
    if date_header_tuple:
        # Parse to UTC datetime
        date_timestamp = mktime_tz(date_header_tuple)

        date_datetime: Optional[datetime] = datetime.fromtimestamp(date_timestamp)
        if date_datetime:
            return date_datetime

    return None


def get_message_subject(subject_header_str: str) -> Optional[str]:
    """
    Get the message subject header as a cleaned string.

    READING:
        * List of potential tags
        https://en.wikipedia.org/wiki/List_of_email_subject_abbreviations

    :param subject_header_str: the message subject header as a string
    :return: clean subject
    """
    if not subject_header_str:
        print(f"Header Error - No 'subject' header: {subject_header_str}")
        return None

    # Remove tagging prefixes such as 'Re' and 'Fwd' using regex.
    subject = str(SUBJECT_PREFIX.sub("", subject_header_str))

    if not ensure_language_english(text=subject):
        return None

    return subject


def get_message_message_id(message_id_str: str) -> str:
    """
    Get the message message-id header as a string.

    NOTE: No need to use unquote, as policy strict bakes this in.

    :param message_id_str: the message message id header as a string
    :return: parsed or generated message id
    """
    # Create message-id if non found
    if not message_id_str:
        message_id_str = make_msgid()

    return message_id_str
