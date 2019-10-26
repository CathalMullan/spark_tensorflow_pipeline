"""
Convert a single message content to a eml file with original appended/not appended.
"""
from email.message import EmailMessage

from nlp_emails.helpers.output.output_eml import output_eml
from nlp_emails.tasks.extract_message_contents import MessageContent

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


def test_output_eml() -> None:
    """
    Convert a single message content to a eml file with original appended/not appended.

    :return: None
    """
    output_eml(message_contents=[MESSAGE_CONTENTS], append_original=False)
    output_eml(message_contents=[MESSAGE_CONTENTS], append_original=True)
