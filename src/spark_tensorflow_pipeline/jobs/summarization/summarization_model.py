"""
Pipeline for text summarization.

See:
    - https://github.com/chen0040/keras-text-summarization
    - https://keras.io/examples/lstm_seq2seq/
    - https://keras.io/examples/lstm_seq2seq_restore/
"""
import math
from datetime import datetime, timedelta
from multiprocessing import Pool
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
from tqdm import tqdm

from spark_tensorflow_pipeline.helpers.config.get_config import CONFIG
from spark_tensorflow_pipeline.jobs.summarization.summarization_config import Seq2SeqConfig, fit_text
from spark_tensorflow_pipeline.processing.vectorize_emails import text_lemmatize_and_lower

FILESYSTEM = gcsfs.GCSFileSystem()

LOCAL_MODEL_WEIGHTS = f"{CONFIG.bucket_summarization_model}seq2seq-weights.h5"
LOCAL_MODEL_CONFIG = f"{CONFIG.bucket_summarization_model}seq2seq-config.npy"
LOCAL_MODEL_ARCHITECTURE = f"{CONFIG.bucket_summarization_model}seq2seq-architecture"


class Seq2SeqSummarizer:
    """
    Interface to run Seq2Seq model.
    """

    model_name = "seq2seq"
    hidden_units = 256

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
        try:
            optimizer = RMSprop(1.0 * hvd.size())
            optimizer = hvd.DistributedOptimizer(optimizer)
        except ValueError:
            print("Running outside Horovod.")
            optimizer = RMSprop(1.0)

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

    def fit(
        self,
        body_train: List[str],
        subject_train: List[str],
        body_test: List[str],
        subject_test: List[str],
        epochs: int,
        batch_size: int,
    ) -> None:
        """
        Begin training of Seq2Seq model, saving trained model in directory provided.

        :param body_train: subset of body data for training
        :param subject_train: subset of subject data for training
        :param body_test: subset of body data for testing
        :param subject_test: subset of subject data for testing
        :param epochs: number of training epochs
        :param batch_size: dataset batching size
        :return: None
        """
        callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if not CONFIG.is_dev:
            if hvd.rank() == 0:
                callbacks.append(ModelCheckpoint(LOCAL_MODEL_WEIGHTS))
                self.model.save_weights(LOCAL_MODEL_WEIGHTS)
                self.model.save(LOCAL_MODEL_ARCHITECTURE, save_format="h5")

                with open(LOCAL_MODEL_CONFIG, "w") as file:
                    file.write(str(self.config))
        else:
            callbacks.append(ModelCheckpoint(LOCAL_MODEL_WEIGHTS))
            self.model.save_weights(LOCAL_MODEL_WEIGHTS)
            self.model.save(LOCAL_MODEL_ARCHITECTURE, save_format="h5")

            with open(LOCAL_MODEL_CONFIG, "w") as file:
                file.write(str(self.config))

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
        subject_sequence[0, 0, self.subject_word_to_index["START_TAG"]] = 1

        subject_text: str = ""
        subject_text_len: int = 0

        terminated = False
        while not terminated:
            subject_tokens, hidden_state, cell_state = self.decoder_model.predict([subject_sequence] + states_value)

            subject_token_index: int = int(np.argmax(subject_tokens[0, -1, :]))
            subject_word: str = self.subject_index_to_word[subject_token_index]
            subject_text_len += 1

            if subject_word not in ("START_TAG", "END_TAG"):
                subject_text += f" {subject_word}"

            if subject_word == "END_TAG" or subject_text_len >= self.max_subject_length:
                terminated = True

            subject_sequence = np.zeros((1, 1, self.subject_count))
            subject_sequence[0, 0, subject_token_index] = 1

            states_value = [hidden_state, cell_state]

        return subject_text.strip()


def join_array_map(text: List[str]) -> str:
    """
    Combine list of strings into a string.

    :param text: list of strings
    :return: concatenated string
    """
    return " ".join(map(str, text))


def load_data() -> Tuple[List[str], List[str]]:
    """
    Gather data from Google Storage.

    :return: two lists of subject and body data
    """
    # Either process yesterdays data or a specific date.
    if not CONFIG.process_date:
        yesterdays_date = datetime.strftime(datetime.now() - timedelta(1), "%Y-%m-%d")
        data_source = f"{CONFIG.bucket_parquet}processed_date={yesterdays_date}/"
    else:
        data_source = f"{CONFIG.bucket_parquet}processed_date={CONFIG.process_date}"

    parquet_data_frames: List[DataFrame] = []

    if not CONFIG.is_dev:
        parquet_files: List[str] = FILESYSTEM.find(data_source)
        for parquet_file in parquet_files[1:]:
            parquet_data_frames.append(pandas.read_parquet(path=f"gs://{parquet_file}", columns=["subject", "body"]))
    else:
        parquet_data_frames.append(
            pandas.read_parquet(
                path=f"{CONFIG.bucket_parquet}parquet/processed_enron.parquet.snappy", columns=["subject", "body"]
            )
        )

    parquet_data_frame: DataFrame = pandas.concat(parquet_data_frames)
    data_frame_length: int = len(parquet_data_frame)

    # Replace empty strings with nan then drop the rows.
    with Pool(processes=12) as pool:
        parquet_data_frame["clean_subject"] = list(
            tqdm(pool.imap(text_lemmatize_and_lower, parquet_data_frame["subject"]), total=data_frame_length)
        )

        parquet_data_frame["clean_body"] = list(
            tqdm(pool.imap(text_lemmatize_and_lower, parquet_data_frame["body"]), total=data_frame_length)
        )

        parquet_data_frame["processed_subject"] = list(
            tqdm(pool.imap(join_array_map, parquet_data_frame["clean_subject"]), total=data_frame_length)
        )

        parquet_data_frame["processed_body"] = list(
            tqdm(pool.imap(join_array_map, parquet_data_frame["clean_body"]), total=data_frame_length)
        )

    parquet_data_frame = parquet_data_frame.replace(r"^\s*$", nan, regex=True).dropna()

    parquet_data_frame = parquet_data_frame[parquet_data_frame["subject"].map(len) > 4]
    parquet_data_frame = parquet_data_frame[parquet_data_frame["subject"].map(len) < 100]

    parquet_data_frame = parquet_data_frame[parquet_data_frame["body"].map(len) > 20]
    parquet_data_frame = parquet_data_frame[parquet_data_frame["body"].map(len) < 500]

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

    if not CONFIG.is_dev:
        if tf.io.gfile.exists(LOCAL_MODEL_WEIGHTS):
            summarizer.load_weights(weight_file_path=LOCAL_MODEL_WEIGHTS)
    else:
        Path(CONFIG.bucket_summarization_model).mkdir(parents=True, exist_ok=True)
        if Path(LOCAL_MODEL_WEIGHTS).exists():
            summarizer.load_weights(weight_file_path=LOCAL_MODEL_WEIGHTS)

    body_train, body_test, subject_train, subject_test = train_test_split(body_list, subject_list, test_size=0.2)

    print("Starting training.")
    summarizer.fit(
        body_train=body_train,
        subject_train=subject_train,
        body_test=body_test,
        subject_test=subject_test,
        epochs=int(math.ceil(100 / hvd.size())),
        batch_size=128,
    )


if __name__ == "__main__":
    main()
