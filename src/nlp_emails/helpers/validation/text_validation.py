"""
Validation of generic text.
"""
import cld3
from bs4 import BeautifulSoup


def ensure_language_english(text: str) -> bool:
    """
    Use cld3 to detect language of text.

    :param text: text string to identify language
    :return: bool whether text is English or not
    """
    # Not enough characters to be certain of language.
    if len(text) < 20:
        return True

    lang_prediction = cld3.get_language(text)

    if lang_prediction.language != "en" and lang_prediction.probability >= 0.9:
        print(f"Text Error - Invalid language: {lang_prediction.language}")
        return False

    return True


def strip_html_contents(text: str) -> str:
    """
    Strip HTML tags from text, returning its contents.

    :param text: string containing HTML
    :return: stripped string with HTML removed
    """
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = str(soup.get_text())

    return stripped_text


def is_valid_length(text: str, minimum: int, maximum: int) -> bool:
    """
    Ensure text is between a min and max.

    :param text: text to measure length
    :param minimum: lower bound integer
    :param maximum: higher bound integer
    :return: bool if valid
    """
    text_len = len(text)
    if text_len >= maximum or text_len <= minimum:
        return True

    return False
