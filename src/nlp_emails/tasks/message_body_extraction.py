"""
Functions to extract contents from message body.
"""
from email.message import EmailMessage
from typing import Optional

from talon import quotations

from nlp_emails.helpers.text_validation import ensure_language_english, strip_html_contents


def get_message_body(message: EmailMessage) -> Optional[str]:
    """
    Get the core message body part as a cleaned string.

    In terms of the goal of the project, we are interested in unique unstructured text of a reasonable length.
    So we discard forwarded/replied emails and those which are too short/long.

    NOTE: As part of the identification of actionable emails, the above is subject to change, but for now the
    presumption is otherwise.

    The Enron dataset used for testing seems to have quite a number of poorly parsed email bodies.
    Issues such as incorrect splitting of urls across lines cause havoc with out attempts to clean the data.
    Will have to make due for now, but some of the 'strange' parsing seem here is a direct result of the data used.

    READING:
        * General discussion on the topic.
        https://en.wikipedia.org/wiki/Posting_style

        * MailGun sited these papers for their 'Talon' project.
        http://www.cs.cmu.edu/~vitor/papers/sigFilePaper_finalversion.pdf
        http://www.cs.cornell.edu/people/tj/publications/joachims_01a.pdf

    :param message: a parsed EmailMessage
    :return: cleaned message body as a string
    """
    core_message: Optional[EmailMessage] = message.get_body()  # type: ignore

    if not isinstance(core_message, EmailMessage):
        return None

    try:
        message_contents = core_message.get_content()
    except (KeyError, LookupError):
        print(f"Body Error - Cannot parse message contents")
        return None

    if core_message.get_content_subtype() == "html":
        message_contents = strip_html_contents(text=message_contents)

    # Use Talon to attempt to remove message quotations
    message_body = str(quotations.extract_from_plain(message_contents))

    if not ensure_language_english(text=message_body):
        return None

    # Avoid large emails which will clog up the pipeline or tiny emails which provide little benefit.
    body_len = len(message_body)
    if body_len > 10_000 or body_len < 200:
        print(f"Body error: Body discarded due to length - {body_len}")
        return None

    return message_body