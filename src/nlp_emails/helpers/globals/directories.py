"""
Directory traversal helper.
"""
from os.path import abspath, dirname

PROJECT_DIR = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))

CONFIG_DIR = PROJECT_DIR + "/config"

DATA_DIR = PROJECT_DIR + "/data"
RAW_DIR = DATA_DIR + "/raw"
ENRON_DIR = RAW_DIR + "/enron"

PROCESSED_DIR = DATA_DIR + "/processed"
PARQUET_DIR = PROJECT_DIR + "/parquet"
CLEAN_ENRON_DIR = PROCESSED_DIR + "/clean_enron"

TESTS_DIR = PROJECT_DIR + "/tests"
TESTS_DATA_DIR = TESTS_DIR + "/data"
TESTS_EMAIL_DIR = TESTS_DATA_DIR + "/emails"
