"""
Directory traversal helper.
"""
from os.path import abspath, dirname

PROJECT_DIR = dirname(dirname(dirname(dirname(abspath(__file__)))))
DATA_DIR = PROJECT_DIR + "/data"
RAW_DIR = DATA_DIR + "/raw"
ENRON_DIR = RAW_DIR + "/enron"
