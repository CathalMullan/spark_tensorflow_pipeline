"""
Test - text_validation.py.
"""
import pytest

from nlp_emails.helpers.validation.text_validation import ensure_language_english, strip_html_contents

SENTENCE = "Here is a string."
ADDRESS = "123 Main St. Nashville, TN 37212"

HTML_TEXT = f"""
<table>
    <tr>
        <td align="center"><img src="logo.jpg"></td>
    </tr>
    <tr>
        <td>{SENTENCE}</td>
    </tr>
    <tr>
        <td align="center">{ADDRESS}</td>
    </tr>
</table>
"""


@pytest.mark.parametrize(
    "sentence, is_english",
    [
        ("This is an English sentence", True),
        ("This is a mostly English sentence - Auf Wiedersehen!", True),
        ("Это русская фраза", True),
        ("Dit is een Nederlandse verklaring", False),
    ],
)
def test_ensure_language_english(sentence: str, is_english: bool) -> None:
    """
    Test if the English detection works correctly. Also verify short text is ignored from checking.

    :param sentence: a sentence in a given language
    :param is_english: whether the core language of the text is English or not
    :return: None
    """
    bool_result = ensure_language_english(text=sentence)
    assert bool_result is is_english


def test_strip_html_contents() -> None:
    """
    Test the use of bs4 to strip HTML tags. Ensure we get the important text stored with the HTML.

    :return: None
    """
    cleaned_text = strip_html_contents(text=HTML_TEXT)

    # Ensure no HTML chars are in the cleaned text
    html_chars = ["<", ">"]
    assert any(char in cleaned_text for char in html_chars) is False

    # Ensure correct strings extracted from HTML
    contents = [SENTENCE, ADDRESS]
    assert all(string in cleaned_text for string in contents) is True
