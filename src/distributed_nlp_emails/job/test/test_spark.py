"""
Usage of Python dependencies within a Spark cluster.
"""
from typing import List

import requests
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import IntegerType, Row
from pyspark.sql.udf import UserDefinedFunction


def main() -> None:
    """
    Usage of Python dependencies within a Spark cluster.

    :return: None
    """
    spark: SparkSession = SparkSession.builder.appName("pyspark_poetry").getOrCreate()

    # Set the logging output to INFO level for verbosity.
    spark.sparkContext.setLogLevel("INFO")

    # Access the JVM logging context.
    jvm_logger = spark.sparkContext._jvm.org.apache.log4j
    logger = jvm_logger.LogManager.getLogger(__name__)
    logger.info("Beginning PySpark job.")

    # List of sites to hit.
    sites: List[str] = [
        "https://www.facebook.com/",
        "https://www.twitter.com/",
        "https://www.github.com/",
        "https://www.google.com/",
    ]

    # Convert list of strings to a distributed PySpark data frame with column 'site'.
    distributed_sites: DataFrame = spark.createDataFrame(list(map(lambda site: Row(site=site), sites)))  # type: ignore
    distributed_sites.show(truncate=False)
    # +-------------------------+
    # |site                     |
    # +-------------------------+
    # |https://www.facebook.com/|
    # |https://www.twitter.com/ |
    # |https://www.github.com/  |
    # |https://www.google.com/  |
    # +-------------------------+

    # User defined function (UDF) to make request and get length of raw response.
    response_length: UserDefinedFunction = F.udf(lambda site: len(requests.get(site).content), IntegerType())

    # Apply the UDF to each row, saving the result in a new column 'length'.
    distributed_sites_with_length: DataFrame = distributed_sites.withColumn("length", response_length("site"))
    distributed_sites_with_length.show(truncate=False)
    # +-------------------------+------+
    # |site                     |length|
    # +-------------------------+------+
    # |https://www.facebook.com/|126440|
    # |https://www.twitter.com/ |311997|
    # |https://www.github.com/  |126192|
    # |https://www.google.com/  |12521 |
    # +-------------------------+------+

    # Collect all the results and sum the total length of all sites.
    total_length: int = distributed_sites_with_length.agg(F.sum("length")).first()[0]
    logger.info(f"The combined length of all sites is {total_length}.")

    spark.stop()
