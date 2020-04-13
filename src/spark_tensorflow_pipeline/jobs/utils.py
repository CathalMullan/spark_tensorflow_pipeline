"""
General machine learning utility functions.
"""
from typing import Dict

from spark_tensorflow_pipeline.helpers.config.get_config import CONFIG


def build_word_to_index_dictionary() -> Dict[str, int]:
    """
    Read static dictionary text file and create vocabulary of terms.

    :return:
    """
    dictionary_set = sorted(set(line.strip().lower() for line in open(CONFIG.dictionary_path)))
    dictionary_dict = {value: key for key, value in enumerate(dictionary_set)}
    return dictionary_dict


def build_index_to_word_dictionary() -> Dict[int, str]:
    """
    Read static dictionary text file and create vocabulary of terms.

    :return:
    """
    dictionary_set = sorted(set(line.strip().lower() for line in open(CONFIG.dictionary_path)))
    # pylint: disable=unnecessary-comprehension
    dictionary_dict = {key: value for key, value in enumerate(dictionary_set)}
    return dictionary_dict


WORD_TO_INDEX_DICTIONARY = build_word_to_index_dictionary()
INDEX_TO_WORD_DICTIONARY = build_index_to_word_dictionary()
