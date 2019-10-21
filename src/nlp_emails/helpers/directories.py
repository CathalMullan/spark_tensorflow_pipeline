"""
Directory traversal helper.
"""
from os.path import abspath, dirname

PROJECT_DIR = dirname(dirname(dirname(dirname(abspath(__file__)))))

DATA_DIR = PROJECT_DIR + "/data"
RAW_DIR = DATA_DIR + "/raw"
ENRON_DIR = RAW_DIR + "/enron"

TESTS_DIR = PROJECT_DIR + "/tests"
TESTS_DATA_DIR = TESTS_DIR + "/data"
TESTS_EMAIL_DIR = TESTS_DATA_DIR + "/emails"
