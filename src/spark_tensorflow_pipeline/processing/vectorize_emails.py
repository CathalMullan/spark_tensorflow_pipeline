"""
Read in Parquet file containing processed eml data, and vectorize to Numpy arrays.
"""
import string
from typing import List, Optional

import spacy
from spacy.tokens.token import Token

from spark_tensorflow_pipeline.jobs.utils import WORD_TO_INDEX_DICTIONARY

# https://blog.dominodatalab.com/making-pyspark-work-spacy-overcoming-serialization-errors/
# spaCy isn't serializable but loading it is semi-expensive

SPACY = spacy.load("en_core_web_sm")


def is_valid_token(token: Token) -> bool:
    """
    Verify a token is fix for our topic detection purposes.

    :param token: a spaCy token
    :return: bool if valid
    """
    if token.like_num or token.like_email or token.like_url:
        return False

    if len(token.pos_) <= 2 or len(token.lemma_) <= 2:
        return False

    if token.lemma_.lower() not in WORD_TO_INDEX_DICTIONARY.keys():
        return False

    return True


def text_lemmatize_and_lower(text: Optional[str]) -> List[str]:
    """
    Remove unwanted characters from text, using spaCy and it's part of speech tagging.

    Strip punctuation and stop words.
    Convert words to their root form.

    :param text: dirty text to be lemmatized
    :return: text cleaned of unwanted characters, lemmatized and lowered
    """
    if not text:
        return [""]

    # Remove punctuation using C lookup table
    # https://stackoverflow.com/a/266162
    text = text.translate(str.maketrans("", "", string.punctuation))
    text_doc = SPACY(text)

    clean_tokens: List[str] = []
    for token in text_doc:
        if is_valid_token(token):
            clean_tokens.append(token.lemma_.lower())

    return clean_tokens
