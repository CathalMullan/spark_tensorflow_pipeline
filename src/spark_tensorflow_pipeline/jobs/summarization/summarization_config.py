"""
Create Keras config from input data.
"""
from collections import Counter
from typing import Counter as CounterType, Dict, List

from dataclasses import dataclass

MAX_BODY_LENGTH = 5000
MAX_SUBJECT_LENGTH = 50
MAX_BODY_VOCAB = 500_000
MAX_SUBJECT_VOCAB = 20_000


@dataclass
class Seq2SeqConfig:
    """
    Config for Seq2Seq model based on dataset.
    """

    body_word_to_index: Dict[str, int]
    body_index_to_word: Dict[int, str]
    subject_word_to_index: Dict[str, int]
    subject_index_to_word: Dict[int, str]
    body_count: int
    subject_count: int
    max_body_length: int
    max_subject_length: int


def fit_text(body_list: List[str], subject_list: List[str]) -> Seq2SeqConfig:
    """
    Create Keras config from input data.

    :param body_list: list of email bodies.
    :param subject_list: list of email subjects,
    :return: config object with dataset statistics and dictionaries
    """
    body_counter: CounterType[str] = Counter()
    subject_counter: CounterType[str] = Counter()
    max_body_length: int = 0
    max_subject_length: int = 0

    for body in body_list:
        body_words: List[str] = body.split(" ")
        body_length: int = len(body_words)
        if body_length > MAX_BODY_LENGTH:
            continue
        for word in body_words:
            body_counter[word] += 1
        max_body_length = max(max_body_length, body_length)

    for subject in subject_list:
        text: List[str] = f"START {subject.lower()} END".split(" ")
        subject_length: int = len(text)
        if subject_length > MAX_SUBJECT_LENGTH:
            continue
        for word in text:
            subject_counter[word] += 1
        max_subject_length = max(max_subject_length, subject_length)

    body_word_to_index: Dict[str, int] = {}
    for index, word in enumerate(body_counter.most_common(MAX_BODY_VOCAB)):  # type: ignore
        body_word_to_index[word[0]] = index + 2
    body_word_to_index["PAD"] = 0
    body_word_to_index["UNK"] = 1

    body_index_to_word: Dict[int, str] = {index: word for word, index in body_word_to_index.items()}
    # body_index_to_word: Dict[int, str] = dict([(index, word) for word, index in body_word_to_index.items()])

    subject_word_to_index: Dict[str, int] = {}
    for index, word in enumerate(body_counter.most_common(MAX_SUBJECT_LENGTH)):  # type: ignore
        subject_word_to_index[word[0]] = index + 1
    subject_word_to_index["UNK"] = 0

    subject_index_to_word: Dict[int, str] = {index: word for word, index in subject_word_to_index.items()}
    # subject_index_to_word: Dict[int, str] = dict([(index, word) for word, index in subject_word_to_index.items()])

    body_count: int = len(body_word_to_index)
    subject_count: int = len(subject_word_to_index)

    return Seq2SeqConfig(
        body_word_to_index=body_word_to_index,
        body_index_to_word=body_index_to_word,
        subject_word_to_index=subject_word_to_index,
        subject_index_to_word=subject_index_to_word,
        body_count=body_count,
        subject_count=subject_count,
        max_body_length=MAX_BODY_LENGTH,
        max_subject_length=MAX_SUBJECT_LENGTH,
    )
