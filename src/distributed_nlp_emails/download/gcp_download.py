"""
Allows downloading of files from Google Cloud Storage.
"""
import os
import sys
from pathlib import Path

import click
from google.cloud import storage
from spark_tensorflow_pipeline.helpers.config.get_config import CONFIG


@click.command()
@click.argument("bucket_file_name")
@click.argument("bucket_prefix")
@click.argument("storage_path")
def main(bucket_file_name: str, bucket_prefix: str, storage_path: str) -> None:
    """
    Download a file from a Google Cloud bucket.

    \b
    :param bucket_file_name: name of file within bucket
    :param bucket_prefix: prefix of bucket name to insert object
    :param storage_path: path to save bucket file
    :return: None
    """
    _, storage_directory = os.path.split(storage_path)

    # Ensure directory exists
    Path(storage_directory).mkdir(parents=True, exist_ok=True)

    if not CONFIG.gcp_credentials.exists() or not CONFIG.gcp_credentials.is_file():
        print(f"Error: Invalid value for 'gcp_credentials': File '{CONFIG.gcp_credentials}' does not exist.")
        sys.exit(1)

    storage_client = storage.Client.from_service_account_json(CONFIG.gcp_credentials)
    bucket_name = bucket_prefix + "-" + storage_client.project
    print(f"Info: Downloading file '{bucket_file_name}' from bucket '{bucket_name}'.")
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(bucket_file_name)
    blob.download_to_filename(storage_path)

    print(f"Success: File '{bucket_file_name}' downloaded to path '{storage_path}'.")
    sys.exit(0)
