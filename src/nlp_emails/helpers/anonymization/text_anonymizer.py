"""
Identify and anonymize personally identifiable information in text.

https://spacy.io/api/annotation.
https://faker.readthedocs.io/en/master/providers.html
"""
from typing import Callable, Dict

import faker
import spacy
from faker.providers import address, company, currency, internet, person, python

# See https://blog.dominodatalab.com/making-pyspark-work-spacy-overcoming-serialization-errors/
SPACY_TAGGER = spacy.load("en_core_web_lg")

FAKE = faker.Faker(locale="en_US")
FAKE.add_provider([internet, person, company, currency, address, python])

FAKER_DICT: Dict[str, Callable[[], str]] = {
    # People, including fictional.
    "PERSON": lambda: str(FAKE.first_name()),
    # Nationalities or religious or political groups.
    "NORP": lambda: str(FAKE.country()),
    # Buildings, airports, highways, bridges, etc.
    "FAC": lambda: str(FAKE.country()),
    # Companies, agencies, institutions, etc.
    "ORG": lambda: str(FAKE.company()),
    # Countries, cities, states.
    "GPE": lambda: str(FAKE.country()),
    # Non-GPE locations, mountain ranges, bodies of water.
    "LOC": lambda: str(FAKE.country()),
    # Objects, vehicles, foods, etc. (Not services.)
    "PRODUCT": lambda: str(FAKE.company()),
    # Named hurricanes, battles, wars, sports events, etc.
    "EVENT": lambda: str(FAKE.name()),
    # Titles of books, songs, etc.
    "WORK_OF_ART": lambda: str(FAKE.name()),
    # Named documents made into laws.
    "LAW": lambda: str(FAKE.name()),
    # Any named language.
    "LANGUAGE": lambda: str(FAKE.name()),
    # Absolute or relative dates or periods.
    "DATE": lambda: str(FAKE.month_name()),
    # Times smaller than a day.
    "TIME": lambda: str(FAKE.time(pattern="%H:%M:%S", end_datetime=None)),
    # Percentage, including ”%“.
    "PERCENT": lambda: str(FAKE.name()),
    # Monetary values, including unit.
    "MONEY": lambda: str(FAKE.currency_name()),
    # Measurements, as of weight or distance.
    "QUANTITY": lambda: str(FAKE.pyint()),
    # “first”, “second”, etc.
    "ORDINAL": lambda: str(FAKE.pyint()),
    # Numerals that do not fall under another type.
    "CARDINAL": lambda: str(FAKE.pyint()),
    # Does the token resemble an email address?
    "EMAIL": lambda: str(FAKE.safe_email()),
    # Does the token resemble a URL?
    "URL": lambda: str(FAKE.url()),
    # Does the token represent a number? e.g. “10.9”, “10”, “ten”, etc.
    "NUM": lambda: str(FAKE.pyint()),
}


def spacy_anonymize_text(text: str) -> str:
    """
    Use spaCy to find and replace entities with their tag.

    :param text: text to be cleaned
    :return: text with entities removed
    """
    doc = SPACY_TAGGER(text)

    for entity in sorted(doc.ents, key=lambda ent: len(ent.text), reverse=True):
        text = text.replace(str(entity.text).strip(), f"{entity.label_}")

    for token in doc:
        if token.like_email:
            text = text.replace(str(token.text).strip(), f"EMAIL")
        elif token.like_url:
            text = text.replace(str(token.text).strip(), f"URL")
        elif token.like_num:
            text = text.replace(str(token.text).strip(), f"NUM")

    return text


def faker_generate_replacements(text: str) -> str:
    """
    Replace tags in text with generated fake data.

    :param text:
    :return:
    """
    parsed_text = SPACY_TAGGER(text, disable=["parser", "tagger", "ner"])
    text_tokens = [token.text for token in parsed_text]

    spacy_tags = list(FAKER_DICT.keys())
    for tag in spacy_tags:
        if tag in text_tokens:
            text = text.replace(tag, FAKER_DICT[tag](), 1)

    return text
