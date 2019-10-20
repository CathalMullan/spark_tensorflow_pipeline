"""
Test.
"""
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional

from nlp_emails.helpers.directories import TESTS_EMAIL_DIR
from nlp_emails.tasks.parse_email_messages import list_files_in_folder, parse_email_messages, read_msg_file


def test_list_files_in_folder() -> None:
    """
    Ensure returns only files.

    :return: None
    """
    file_paths: List[Path] = list_files_in_folder(folder_path=TESTS_EMAIL_DIR)
    for file_path in file_paths:
        assert file_path.is_file()


def test_read_msg_file() -> None:
    """
    Ensure parsing can handle different charsets/faulty emails.

    :return: None
    """
    file_paths: List[Path] = list_files_in_folder(folder_path=TESTS_EMAIL_DIR)
    for file_path in file_paths:
        email_message: Optional[EmailMessage] = read_msg_file(eml_path=file_path)
        if email_message is not None:
            assert isinstance(email_message, EmailMessage)


def test_parse_email_messages() -> None:
    """
    All in one test of parse_email_messages.py.

    :return: None
    """
    parse_email_messages(folder_path=TESTS_EMAIL_DIR)
