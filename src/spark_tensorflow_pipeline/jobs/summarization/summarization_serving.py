"""
Load and serve summarization model.
"""
from typing import Any, Dict

import numpy as np

from spark_tensorflow_pipeline.helpers.globals.directories import MODELS_DIR
from spark_tensorflow_pipeline.jobs.summarization.summarization_config import Seq2SeqConfig
from spark_tensorflow_pipeline.jobs.summarization.summarization_model import Seq2SeqSummarizer, load_data


def main() -> None:
    """
    Load and serve summarization model.

    :return: None
    """
    print("Loading input data.")
    subject_list, body_list = load_data()

    config_file: str = Seq2SeqSummarizer.get_config_file_path(model_dir_path=f"{MODELS_DIR}/summarization")
    # noinspection PyTypeChecker
    raw_config: Dict[str, Any] = np.load(config_file, allow_pickle=True).item()  # type: ignore

    config: Seq2SeqConfig = Seq2SeqConfig(**raw_config)
    summarizer: Seq2SeqSummarizer = Seq2SeqSummarizer(config)

    weight_file: str = Seq2SeqSummarizer.get_weight_file_path(model_dir_path=f"{MODELS_DIR}/summarization")
    summarizer.load_weights(weight_file_path=weight_file)

    print("Starting predicting.")
    for index, _ in enumerate(body_list):
        print(f"Generated Subject: {summarizer.summarize(body_list[index])}")
        print(f"Expected Subject: {subject_list[index]}")


if __name__ == "__main__":
    main()
