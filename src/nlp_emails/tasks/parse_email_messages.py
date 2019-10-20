"""
Load the Enron dataset into MIME messages.
"""
from email import message_from_file
from email.errors import MessageDefect
from email.message import EmailMessage
from email.policy import strict
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional

from nlp_emails.helpers.directories import ENRON_DIR


def list_files_in_folder(folder_path: Path) -> List[Path]:
    """
    List all files in a folder recursively.

    :param folder_path: starting point of search
    :return: list of paths to files
    """
    all_paths = list(folder_path.rglob(f"**/*"))

    # Strip non-files
    file_paths: List[Path] = [path for path in all_paths if path.is_file()]

    return file_paths


def read_msg_file(eml_path: Path) -> Optional[EmailMessage]:
    """
    Open a msg file and read its contents, parses to EmailMessage.

    :param eml_path: Path to an eml file
    :return: parsed EmailMessage from file
    """
    with eml_path.open(encoding="utf-8", errors="replace") as file:
        try:
            # the policy 'strict' makes this return an EmailMessage class (Python 3.6+), rather than a Message class.
            email_message: EmailMessage = message_from_file(file, policy=strict)  # type: ignore

            # Include additional headers for debugging purposes
            email_message.add_header("x-file-path", file.name)

            return email_message
        except MessageDefect:
            print(f"Could not parse file {str(eml_path)} to EmailMessage")
            return None


def parse_email_messages(folder_path: Path = Path(f"{ENRON_DIR}/maildir")) -> List[EmailMessage]:
    """
    Read all files in a directory recursively and parse to EmailMessages.

    NOTE: Default is to read from the Enron dataset, which contain over 500,000 eml files.

    :param folder_path: folder containing eml files
    :return: list of parsed email messages from file contents
    """
    file_paths: List[Path] = list_files_in_folder(folder_path)

    # DEBUG - Sort files like Unix filesystem
    # str_paths: List[str] = [str(path) for path in file_paths]
    # str_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    # DEBUG - Limit number of files processed
    # file_paths = [Path(path) for path in str_paths]
    # file_paths = file_paths[:100_000]

    print(f"Total number of files in directory: {len(file_paths)}")
    with Pool(processes=8) as pool:
        potential_messages: List[Optional[EmailMessage]] = pool.map(read_msg_file, file_paths)

    # Strip out mails which failed in parsing
    email_messages: List[EmailMessage] = [msg for msg in potential_messages if msg is not None]
    print(f"Total number of eml files successfully parsed: {len(email_messages)}")

    return email_messages
