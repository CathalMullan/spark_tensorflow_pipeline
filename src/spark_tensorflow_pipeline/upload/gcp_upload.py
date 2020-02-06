"""
Allows uploading of files to Google Cloud Storage.
"""
import sys
from pathlib import Path

import click
from google.api_core.exceptions import Conflict
from google.cloud import storage

from spark_tensorflow_pipeline.helpers.config.get_config import CONFIG


@click.command()
@click.argument("local_file", type=click.Path(exists=True))
@click.argument("bucket_prefix")
def main(local_file: str, bucket_prefix: str) -> None:
    """
    Upload a file to a Google Cloud bucket.

    \b
    :param local_file: local file to upload
    :param bucket_prefix: prefix of bucket name to insert object
    :return: None
    """
    local_file_path = Path(local_file)
    if not local_file_path.exists() or not local_file_path.is_file():
        print(f"Error: Invalid value for 'local_file': File '{local_file}' does not exist.")
        sys.exit(1)

    if not CONFIG.gcp_credentials.exists() or not CONFIG.gcp_credentials.is_file():
        print(f"Error: Invalid value for 'gcp_credentials': File '{CONFIG.gcp_credentials}' does not exist.")
        sys.exit(1)

    storage_client = storage.Client.from_service_account_json(CONFIG.gcp_credentials)
    bucket_name = bucket_prefix + "-" + storage_client.project
    print(f"Info: Storing file '{local_file_path.name}' in bucket '{bucket_name}'.")
    bucket = storage_client.bucket(bucket_name)

    try:
        bucket.create()
    except Conflict:
        print(f"Info: Bucket '{bucket_name}' already exists.")

    blob = bucket.blob(local_file_path.name)
    blob.upload_from_filename(local_file)

    print(f"Success: File '{local_file}' uploaded to bucket '{bucket_name}'.")
    sys.exit(0)
