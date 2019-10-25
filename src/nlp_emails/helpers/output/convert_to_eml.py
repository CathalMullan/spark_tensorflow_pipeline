"""
Convert a list of message contents into a eml files.
"""
import time
from pathlib import Path
from typing import List

from nlp_emails.helpers.globals.directories import CLEAN_ENRON_DIR
from nlp_emails.tasks.extract_message_contents import MessageContent


def convert_to_eml(message_contents: List[MessageContent]) -> None:
    """
    Convert a list of message contents into eml files and save to processed clean Enron directory.

    :param message_contents: list of parsed message contents
    :return:
    """
    Path(CLEAN_ENRON_DIR).mkdir(exist_ok=True)

    for message_content in message_contents:
        with open(f"{CLEAN_ENRON_DIR}/{int(round(time.time() * 1000))}", "w") as file:
            file.write(message_content.as_str() + "\n")
            file.write("\n")
