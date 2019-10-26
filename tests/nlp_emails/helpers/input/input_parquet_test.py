"""
Read in a basic Parquet file and verify it parses to Pandas.
"""
from pandas import DataFrame

from nlp_emails.helpers.globals.directories import TESTS_PARQUET_DIR
from nlp_emails.helpers.input.input_parquet import read_dataframe_from_parquet


def test_read_dataframe_from_parquet() -> None:
    """
    Read in a basic Parquet file and verify it parses to Pandas.

    :return: None
    """
    pandas_dataframe = read_dataframe_from_parquet(TESTS_PARQUET_DIR + "/test.parquet.snappy")
    assert isinstance(pandas_dataframe, DataFrame)
