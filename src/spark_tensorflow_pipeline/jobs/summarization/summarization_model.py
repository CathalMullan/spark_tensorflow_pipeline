"""
Pipeline for text summarization.

See:
    - https://github.com/chen0040/keras-text-summarization
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from spark_tensorflow_pipeline.jobs.summarization.fake_news_loader import fit_text
from spark_tensorflow_pipeline.jobs.summarization.plot_utils import plot_and_save_history
from spark_tensorflow_pipeline.jobs.summarization.seq2seq import Seq2SeqSummarizer
from spark_tensorflow_pipeline.helpers.globals.directories import PROJECT_DIR
LOAD_EXISTING_WEIGHTS = False


def main():
    """

    :return:
    """
    data_dir_path = PROJECT_DIR + "/data/summarization"
    report_dir_path = PROJECT_DIR + "/data"
    model_dir_path = PROJECT_DIR + "/models"

    print("loading csv file ...")
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    print("extract configuration from input texts ...")
    y = df.title
    x = df["text"]

    config = fit_text(x, y)
    summarizer = Seq2SeqSummarizer(config)

    if LOAD_EXISTING_WEIGHTS:
        summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print("demo size: ", len(x_train))
    print("testing size: ", len(x_test))

    print("start fitting ...")
    history = summarizer.fit(x_train, y_train, x_test, y_test, epochs=100)

    history_plot_file_path = report_dir_path + "/" + Seq2SeqSummarizer.model_name + "-history.png"
    if LOAD_EXISTING_WEIGHTS:
        history_plot_file_path = (
            report_dir_path + "/" + Seq2SeqSummarizer.model_name + "-history-v" + str(summarizer.version) + ".png"
        )
    plot_and_save_history(history, summarizer.model_name, history_plot_file_path, metrics={"loss", "acc"})


if __name__ == "__main__":
    main()
