"""
End to end processing on the Enron dataset.
"""
import time
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Optional

from nlp_emails.helpers.globals.directories import ENRON_DIR
from nlp_emails.helpers.output.convert_to_eml import convert_to_eml
from nlp_emails.helpers.output.convert_to_parquet import convert_to_parquet
from nlp_emails.tasks.extract_message_contents import MessageContent, eml_path_to_message_contents
from nlp_emails.tasks.parse_email_messages import list_files_in_folder


def enron_pipeline() -> None:
    """
    Read in the Enron dataset, parse out contents while anonymizing them then save to an parquet file and eml files.

    Count: 8257
    Finish: 117 seconds (~2 minutes)

    Count: 459629
    Finish: 5989 seconds (~100 minutes)

    :return:
    """
    file_paths: List[Path] = list_files_in_folder(f"{ENRON_DIR}/maildir")

    print("Start")
    start_time: int = int(time.time())

    pool = Pool(processes=8)
    try:
        optional_message_contents: List[Optional[MessageContent]] = pool.map(eml_path_to_message_contents, file_paths)
    finally:
        pool.close()
        pool.join()

    message_contents: List[MessageContent] = [message for message in optional_message_contents if message]

    convert_to_eml(message_contents)
    convert_to_parquet(message_contents)

    print(f"Count: {len(message_contents)}")
    print(f"Finish: {int(time.time()) - start_time} seconds", flush=True)


if __name__ == "__main__":
    enron_pipeline()
