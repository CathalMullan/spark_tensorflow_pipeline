"""

"""
import pandas as pd
import numpy as np

from spark_tensorflow_pipeline.jobs.summarization.seq2seq import Seq2SeqSummarizer


def main():
    """

    :return:
    """
    data_dir_path = './data'
    model_dir_path = './models'

    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")
    x = df['text']
    y = df.title

    config = np.load(Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path)).item()

    summarizer = Seq2SeqSummarizer(config)
    summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    print('start predicting ...')
    for i in np.random.permutation(np.arange(len(x)))[0:20]:
        article = x[i]
        actual_headline = y[i]
        headline = summarizer.summarize(x)
        print('Article: ', article)
        print('Generated Headline: ', headline)
        print('Original Headline: ', actual_headline)


if __name__ == '__main__':
    main()
