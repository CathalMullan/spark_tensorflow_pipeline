"""
Pipeline for text summarization.

See:
    - https://github.com/chen0040/keras-text-summarization\
    - https://keras.io/examples/lstm_seq2seq/
    - https://keras.io/examples/lstm_seq2seq_restore/
"""
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, List, Tuple

import gcsfs
import horovod.tensorflow.keras as hvd
import numpy as np
import pandas
import tensorflow as tf
from numpy import nan
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences

from spark_tensorflow_pipeline.helpers.config.get_config import CONFIG
from spark_tensorflow_pipeline.helpers.globals.directories import MODELS_DIR
from spark_tensorflow_pipeline.jobs.summarization.summarization_config import Seq2SeqConfig, fit_text

FILESYSTEM = gcsfs.GCSFileSystem()


class Seq2SeqSummarizer:
    """
    Interface to run Seq2Seq model.
    """

    model_name = "seq2seq"
    hidden_units = 100

    def __init__(self, config: Seq2SeqConfig):
        """
        Initialize model for training.

        :param config: seq2seq config from input data
        """
        self.body_count = config.body_count
        self.max_body_length = config.max_body_length
        self.subject_count = config.subject_count
        self.max_subject_length = config.max_subject_length
        self.body_word_to_index = config.body_word_to_index
        self.body_index_to_word = config.body_index_to_word
        self.subject_word_to_index = config.subject_word_to_index
        self.subject_index_to_word = config.subject_index_to_word
        self.config = config.__dict__

        encoder_inputs: Input = Input(shape=(None,), name="encoder_inputs")
        encoder_embedding: Embedding = Embedding(
            input_dim=self.body_count,
            output_dim=self.hidden_units,
            input_length=self.max_body_length,
            name="encoder_embedding",
        )
        encoder_lstm: LSTM = LSTM(units=self.hidden_units, return_state=True, name="encoder_lstm")
        _, encoder_hidden_state, encoder_cell_state = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states: List[np.ndarray] = [encoder_hidden_state, encoder_cell_state]

        decoder_inputs: Input = Input(shape=(None, self.subject_count), name="decoder_inputs")
        decoder_lstm: LSTM = LSTM(
            units=self.hidden_units, return_state=True, return_sequences=True, name="decoder_lstm"
        )
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(units=self.subject_count, activation="softmax", name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

        # Horovod: add Horovod Distributed Optimizer.
        optimizer = RMSprop(1.0 * hvd.size())
        optimizer = hvd.DistributedOptimizer(optimizer)

        model: Model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
            experimental_run_tf_function=False,
        )

        self.model = model
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs: List[Input] = [Input(shape=(self.hidden_units,)), Input(shape=(self.hidden_units,))]
        decoder_outputs, hidden_state, cell_state = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states: List[Dense] = [hidden_state, cell_state]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def load_weights(self, weight_file_path: str) -> None:
        """
        Load existing model from weights file.

        :param weight_file_path: path to previous weights
        :return: None
        """
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def transform_body_text(self, text: List[str]) -> np.ndarray:
        """
        Convert 'body' text to vectorized array, handling OOV tokens.

        :param text: list of text
        :return: text as a numpy array matching a static dictionary
        """
        text_list: List[List[int]] = []
        for line in text:
            word_list: List[int] = []
            for word in line.lower().split(" "):
                # Default OOV token [UNK].
                word_id = 1

                if word in self.body_word_to_index:
                    word_id = self.body_word_to_index[word]

                word_list.append(word_id)
                if len(word_list) >= self.max_body_length:
                    break
            text_list.append(word_list)

        text_array: np.ndarray = pad_sequences(text_list, maxlen=self.max_body_length)
        return text_array

    def transform_subject_encoding(self, text: List[str]) -> np.ndarray:
        """
        Convert 'subject' text to vectorized array.

        :param text: list of text
        :return: text as a numpy array matching a static dictionary
        """
        text_list: List[List[str]] = []

        for line in text:
            word_list: List[str] = []
            for word in f"START {line.lower()} END".split(" "):
                word_list.append(word)

                if len(word_list) >= self.max_subject_length:
                    break
            text_list.append(word_list)
        text_array: np.ndarray = np.array(text_list)
        return text_array

    def generate_batch(
        self, body_samples: np.ndarray, subject_samples: np.ndarray, batch_size: int
    ) -> Generator[Tuple[List[np.ndarray], np.ndarray], None, None]:
        """
        Batch data into chunks.

        :param body_samples: body list as a numpy array
        :param subject_samples: subject list as a numpy array
        :param batch_size: size of chunks
        :return: generator yielding encoded/decoded body data with decoded subject
        """
        num_batches: int = len(body_samples) // batch_size

        while True:
            for batch_index in range(0, num_batches):
                start: int = batch_index * batch_size
                end: int = (batch_index + 1) * batch_size

                encoder_body_data_batch: np.ndarray = pad_sequences(
                    sequences=body_samples[start:end], maxlen=self.max_body_length
                )

                decoder_subject_data_batch: np.ndarray = np.zeros(
                    shape=(batch_size, self.max_subject_length, self.subject_count)
                )

                decoder_body_data_batch: np.ndarray = np.zeros(
                    shape=(batch_size, self.max_subject_length, self.subject_count)
                )

                for line_index, subject_words in enumerate(subject_samples[start:end]):
                    for index, word in enumerate(subject_words):
                        # Default OOV token [UNK].
                        word_to_index = 0

                        if word in self.subject_word_to_index:
                            word_to_index = self.subject_word_to_index[word]

                        if word_to_index != 0:
                            decoder_body_data_batch[line_index, index, word_to_index] = 1
                            if index > 0:
                                decoder_subject_data_batch[line_index, index - 1, word_to_index] = 1

                yield [encoder_body_data_batch, decoder_body_data_batch], decoder_subject_data_batch

    @staticmethod
    def get_weight_file_path(model_dir_path: str) -> str:
        """
        Get the path to the weight file.

        :param model_dir_path: path containing model
        :return: path to weight file
        """
        return f"{model_dir_path}/{Seq2SeqSummarizer.model_name}-weights.h5"

    @staticmethod
    def get_config_file_path(model_dir_path: str) -> str:
        """
        Get the path to the config file.

        :param model_dir_path: path containing model
        :return: path to config file
        """
        return f"{model_dir_path}/{Seq2SeqSummarizer.model_name}-config.npy"

    @staticmethod
    def get_architecture_file_path(model_dir_path: str) -> str:
        """
        Get the path to the architecture file.

        :param model_dir_path: path containing model
        :return: path to architecture file
        """
        return f"{model_dir_path}/{Seq2SeqSummarizer.model_name}-architecture.json"

    def fit(
        self,
        body_train: List[str],
        subject_train: List[str],
        body_test: List[str],
        subject_test: List[str],
        epochs: int,
        batch_size: int,
        model_dir_path: str,
    ) -> None:
        """
        Begin training of Seq2Seq model, saving trained model in directory provided.

        :param body_train: subset of body data for training
        :param subject_train: subset of subject data for training
        :param body_test: subset of body data for testing
        :param subject_test: subset of subject data for testing
        :param epochs: number of training epochs
        :param batch_size: dataset batching size
        :param model_dir_path: where to save model
        :return: None
        """
        config_file_path: str = Seq2SeqSummarizer.get_config_file_path(model_dir_path)
        weight_file_path: str = Seq2SeqSummarizer.get_weight_file_path(model_dir_path)

        callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            tf.io.gfile.makedirs(model_dir_path)
            callbacks.append(ModelCheckpoint(weight_file_path))

            np.save(config_file_path, self.config)
            architecture_file_path: str = Seq2SeqSummarizer.get_architecture_file_path(model_dir_path)
            open(architecture_file_path, "w").write(self.model.to_json())

        subject_train_array: np.ndarray = self.transform_subject_encoding(subject_train)
        subject_test_array: np.ndarray = self.transform_subject_encoding(subject_test)

        body_train_array: np.ndarray = self.transform_body_text(body_train)
        body_test_array: np.ndarray = self.transform_body_text(body_test)

        train_generator: Generator[Tuple[List[np.ndarray], np.ndarray], None, None] = self.generate_batch(
            body_samples=body_train_array, subject_samples=subject_train_array, batch_size=batch_size
        )

        test_generator: Generator[Tuple[List[np.ndarray], np.ndarray], None, None] = self.generate_batch(
            body_samples=body_test_array, subject_samples=subject_test_array, batch_size=batch_size
        )

        train_batches: int = len(body_train_array) // batch_size
        test_batches: int = len(body_test_array) // batch_size

        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_batches // hvd.size(),
            epochs=epochs,
            verbose=1 if hvd.rank() == 0 else 0,
            validation_data=test_generator,
            validation_steps=test_batches,
            callbacks=callbacks,
        )

        self.model.save_weights(weight_file_path)

    def summarize(self, body_text: str) -> str:
        """
        Summarize input text data.

        :param body_text: body text to be summarized
        :return: summarized 'subject' of body
        """
        body_sequence: List[List[int]] = []
        body_word_ids: List[int] = []

        for word in body_text.lower().split(" "):
            # Default OOV token [UNK].
            index = 1

            if word in self.body_word_to_index:
                index = self.body_word_to_index[word]

            body_word_ids.append(index)

        body_sequence.append(body_word_ids)
        body_sequence_array: np.ndarray = pad_sequences(body_sequence, self.max_body_length)

        states_value: List[np.ndarray] = self.encoder_model.predict(body_sequence_array)
        subject_sequence: np.ndarray = np.zeros((1, 1, self.subject_count))
        subject_sequence[0, 0, self.subject_word_to_index["START"]] = 1

        subject_text: str = ""
        subject_text_len: int = 0

        terminated = False
        while not terminated:
            subject_tokens, hidden_state, cell_state = self.decoder_model.predict([subject_sequence] + states_value)

            subject_token_index: int = int(np.argmax(subject_tokens[0, -1, :]))
            subject_word: str = self.subject_index_to_word[subject_token_index]
            subject_text_len += 1

            if subject_word not in ("START", "END"):
                subject_text += f" {subject_word}"

            if subject_word == "END" or subject_text_len >= self.max_subject_length:
                terminated = True

            subject_sequence = np.zeros((1, 1, self.subject_count))
            subject_sequence[0, 0, subject_token_index] = 1

            states_value = [hidden_state, cell_state]

        return subject_text.strip()


def load_data() -> Tuple[List[str], List[str]]:
    """
    Gather data from Google Storage.

    :return: two lists of subject and body data
    """
    # Either process yesterdays data or a specific date.
    if not CONFIG.process_date:
        # Use consistent bucket for dev testing.
        if CONFIG.is_dev:
            data_source = f"{CONFIG.bucket_parquet}processed_date=2020-04-03/"
        else:
            yesterdays_date = datetime.strftime(datetime.now() - timedelta(1), "%Y-%m-%d")
            data_source = f"{CONFIG.bucket_parquet}processed_date={yesterdays_date}/"
    else:
        data_source = f"{CONFIG.bucket_parquet}processed_date={CONFIG.process_date}"

    parquet_data_frames: List[DataFrame] = []

    parquet_files: List[str] = FILESYSTEM.find(data_source)
    for parquet_file in parquet_files[1:]:
        parquet_data_frames.append(pandas.read_parquet(path=f"gs://{parquet_file}", columns=["subject", "body"]))

    parquet_data_frame: DataFrame = pandas.concat(parquet_data_frames)

    # Replace empty strings with nan then drop the rows.
    parquet_data_frame = parquet_data_frame.replace(r"^\s*$", nan, regex=True).dropna()
    print(f"Processing: {len(parquet_data_frame)} emails.")

    subject_list: List[str] = parquet_data_frame.get("subject").tolist()
    body_list: List[str] = parquet_data_frame.get("body").tolist()

    return subject_list, body_list


def main() -> None:
    """
    Start training Seq2Seq model.

    :return: None
    """
    # Horovod: initialize Horovod.
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    gpu_list = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpu_list:
        print("Visible GPUs detected.")
        tf.config.experimental.set_visible_devices(gpu_list[hvd.local_rank()], "GPU")

    print("Loading input data.")
    subject_list, body_list = load_data()

    config: Seq2SeqConfig = fit_text(body_list, subject_list)
    summarizer: Seq2SeqSummarizer = Seq2SeqSummarizer(config)

    if Path(f"{MODELS_DIR}/summarization/seq2seq-config.npy").exists():
        summarizer.load_weights(
            weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=f"{MODELS_DIR}/summarization")
        )

    body_train, body_test, subject_train, subject_test = train_test_split(body_list, subject_list, test_size=0.2)

    print("Starting training.")
    summarizer.fit(
        body_train=body_train,
        subject_train=subject_train,
        body_test=body_test,
        subject_test=subject_test,
        epochs=int(math.ceil(100 / hvd.size())),
        batch_size=64,
        model_dir_path=f"{MODELS_DIR}/summarization",
    )


if __name__ == "__main__":
    main()
