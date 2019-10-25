"""
Convert a list of message contents into a .parquet.snappy file.
"""
from pathlib import Path
from typing import List

import pandas as pd

from nlp_emails.helpers.globals.directories import PARQUET_DIR
from nlp_emails.tasks.extract_message_contents import MessageContent


def convert_to_parquet(message_contents: List[MessageContent], file_name: str = "processed_enron") -> None:
    """
    Convert a list of message contents into a .parquet.snappy file.

    :param message_contents: list of message content
    :param file_name: name of file to be saved (prepended with .parquet.gzip)
    :return: None
    """
    output_file = f"{PARQUET_DIR}/{file_name}.parquet.snappy"
    try:
        Path(output_file).touch()
    except OSError:
        return None

    message_contents_dict = [message_content.as_dict() for message_content in message_contents]

    data_frame = pd.DataFrame(message_contents_dict)
    data_frame.to_parquet(fname=output_file, engine="pyarrow", compression="snappy")

    return None
