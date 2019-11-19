"""
Convert a single message content to a parquet file with valid/invalid paths.
"""
from email.message import EmailMessage

from distributed_nlp_emails.helpers.output.output_parquet import output_parquet
from distributed_nlp_emails.tasks.extract_message_contents import MessageContent

MESSAGE_CONTENTS = MessageContent(
    original_message=EmailMessage(),
    message_id="hello@world.com",
    date=None,
    from_address="valid@email_1.com",
    to_address_list=["valid@email_2.com"],
    cc_address_list=None,
    bcc_address_list=None,
    subject="",
    body="Here is a valid body",
)


def test_output_parquet() -> None:
    """
    Convert a single message content to a parquet file directory.

    :return: None
    """
    output_parquet(message_contents=[MESSAGE_CONTENTS], file_name="test")
