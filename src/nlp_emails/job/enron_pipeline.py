"""
End to end processing on the Enron dataset.
"""
import time
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Optional

from nlp_emails.helpers.globals.directories import ENRON_DIR, list_files_in_folder
from nlp_emails.helpers.output.output_eml import output_eml
from nlp_emails.helpers.output.output_parquet import output_parquet
from nlp_emails.tasks.extract_message_contents import MessageContent, eml_path_to_message_contents


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

    # DEBUG - Limit number of files processed
    file_paths = file_paths[:10]

    # DEBUG - Sort files like Unix filesystem
    str_paths: List[str] = [str(path) for path in file_paths]
    str_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    file_paths = [Path(path) for path in str_paths]

    start_time: int = int(time.time())

    pool = Pool(processes=8)
    try:
        optional_message_contents: List[Optional[MessageContent]] = pool.map(eml_path_to_message_contents, file_paths)
    finally:
        pool.close()
        pool.join()

    message_contents: List[MessageContent] = [message for message in optional_message_contents if message]

    output_eml(message_contents, append_original=True)
    output_parquet(message_contents)

    print(f"Count: {len(message_contents)}")
    print(f"Finish: {int(time.time()) - start_time} seconds", flush=True)


if __name__ == "__main__":
    enron_pipeline()
