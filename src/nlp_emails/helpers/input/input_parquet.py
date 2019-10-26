"""
Read a Parquet file into a Pandas Dataframe using Arrow.
"""
from pathlib import Path
from typing import Union

import pyarrow.parquet
from pandas import DataFrame


def read_dataframe_from_parquet(parquet_file: Union[Path, str]) -> DataFrame:
    """
    Read a Parquet file into a Pandas Dataframe using Arrow.

    :param parquet_file: path to a Parquet file
    :return: Pandas DataFrame of Parquet input file
    """
    if not isinstance(parquet_file, Path):
        parquet_file = Path(parquet_file)

    arrow_df = pyarrow.parquet.read_table(parquet_file)

    pandas_df = arrow_df.to_pandas()
    return pandas_df
