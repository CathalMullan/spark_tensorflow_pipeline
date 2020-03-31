"""
Download the spaCy model.
"""
import spacy


def main() -> None:
    """
    Download the spaCy model.

    :return: None
    """
    spacy.cli.download(model="en_core_web_sm")
