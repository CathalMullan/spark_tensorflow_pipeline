"""
Pipeline for topic modelling.

See
    - https://git.io/Jvsyn
"""
import functools
import pickle  # nosec
from typing import Any, Callable, Dict, List, Optional, Tuple

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from horovod import spark
from scipy.sparse import load_npz
from tensorflow_probability import distributions as tfd

from spark_tensorflow_pipeline.helpers.globals.directories import MODELS_DIR, PROCESSED_DIR


class Parameters:
    """
    Parameter settings for TensorFlow init.
    """

    layer_sizes: List[int] = [300, 300, 300]
    learning_rate: float = 3e-4
    max_steps: int = 180_000
    num_topics: int = 50
    activation: str = "relu"
    prior_initial_value: float = 0.7
    prior_burn_in_steps: int = 120_000
    viz_steps: int = 1_000
    model_dir: Optional[str] = f"{MODELS_DIR}/topic_checkpoint"
    saved_model_dir = f"{MODELS_DIR}/topic_model"
    vocabulary: Dict[int, str] = {}
    batch_size: int = 32


def clip_dirichlet_parameters(tensor: tf.Tensor) -> tf.Tensor:
    """
    Clips the Dirichlet parameters to the numerically stable KL region.

    :param tensor: Tensor to be clipped
    :return: clipped Tensor within range
    """
    return tf.clip_by_value(tensor, 1e-3, 1e3)


def make_encoder(activation: str, num_topics: int, layer_sizes: List[int]) -> Callable[[tf.Tensor], tfd.Dirichlet]:
    """
    Create the encoder function.

    :param activation: Activation function to use.
    :param num_topics: The number of topics.
    :param layer_sizes: The number of hidden units per layer in the encoder.
    :return: A callable mapping a bag-of-words Tensor to a tfd.Distribution instance over topics.
    """
    encoder_net = tf.keras.Sequential()
    for num_hidden_units in layer_sizes:
        encoder_net.add(
            tf.keras.layers.Dense(num_hidden_units, activation=activation, kernel_initializer="glorot_normal")
        )

    encoder_net.add(tf.keras.layers.Dense(num_topics, activation=tf.nn.softplus, kernel_initializer="glorot_normal"))

    def encoder(bag_of_words: tf.Tensor) -> tfd.Dirichlet:
        """
        Map bag-of-words to a Dirichlet instance.

        :param bag_of_words: numpy nd array of values
        :return: clipped Dirichlet of dictionary
        """
        net = clip_dirichlet_parameters(encoder_net(bag_of_words))
        return tfd.Dirichlet(concentration=net, name="topics_posterior")

    return encoder


def make_decoder(num_topics: int, num_words: int) -> Tuple[Callable[[tf.Tensor], tfd.OneHotCategorical], tf.Variable]:
    """
    Create the decoder function.

    :param num_topics: The number of topics.
    :param num_words: The number of words.
    :return: A callable mapping a Tensor of encodings to a tfd.Distribution instance over words.
    """
    glorot_normal_initializer = tf.initializers.glorot_normal()

    topics_words_logits = tf.Variable(
        name="topics_words_logits", initial_value=glorot_normal_initializer([num_topics, num_words])
    )

    topics_words = tf.nn.softmax(topics_words_logits, axis=-1)

    def decoder(topics: tf.Tensor) -> tfd.OneHotCategorical:
        """
        Map Tensor to a OneHotCategorical instance.

        :param topics: Tensor containing topic values
        :return: OneHotCategorical of topics
        """
        word_probabilities = tf.matmul(topics, topics_words)

        # The observations are bag of words and therefore not one-hot.
        # However, log_prob of OneHotCategorical computes the probability correctly in this case.
        return tfd.OneHotCategorical(probs=word_probabilities, name="bag_of_words")

    return decoder, topics_words


def make_prior(num_topics: int, initial_value: float) -> Tuple[Callable[[], tfd.Dirichlet], List[tf.Variable]]:
    """
    Create the prior distribution.

    :param num_topics: Number of topics.
    :param initial_value: The starting value for the prior parameters.
    :return:
        - A callable that returns a tf.distribution.Distribution instance, the prior distribution.
        - A list of Variable objects, the trainable parameters of the prior.
    """
    soft_plus_inverse_initializer = tfp.math.softplus_inverse(
        tf.constant(value=initial_value, shape=[1, num_topics], dtype=tf.float32)
    )

    logit_concentration = tf.Variable(name="logit_concentration", initial_value=soft_plus_inverse_initializer)
    concentration = clip_dirichlet_parameters(tf.nn.softplus(logit_concentration))
    prior_variables = [logit_concentration]

    def prior() -> tfd.Dirichlet:
        """
        Create Dirichlet instance for prior distribution.

        :return: Dirichlet instance
        """
        return tfd.Dirichlet(concentration=concentration, name="topics_prior")

    return prior, prior_variables


def model_fn(features: tf.Tensor, mode: tf.estimator.ModeKeys, params: Parameters) -> tf.estimator.EstimatorSpec:
    """
    Build the model function for use in an estimator.

    :param features: The input features for the estimator.
    :param mode: Signifies whether it is train or test or predict.
    :param params: Some hyper-parameters and config.
    :return: A tf.estimator.EstimatorSpec instance.
    """
    encoder = make_encoder(params.activation, params.num_topics, params.layer_sizes)
    decoder, topics_words = make_decoder(params.num_topics, features.shape[1])
    prior, prior_variables = make_prior(params.num_topics, params.prior_initial_value)

    topics_prior = prior()
    alpha = topics_prior.concentration

    topics_posterior = encoder(features)
    topics = topics_posterior.sample()
    random_reconstruction = decoder(topics)

    reconstruction = random_reconstruction.log_prob(features)
    tf.summary.scalar("reconstruction", tf.reduce_mean(input_tensor=reconstruction))

    # Compute the KL-divergence between two Dirichlet values analytically.
    # The sampled KL does not work well for "sparse" distributions
    kl_divergence = tfd.kl_divergence(topics_posterior, topics_prior)
    tf.summary.scalar("kl", tf.reduce_mean(input_tensor=kl_divergence))

    # Ensure that the KL is non-negative (up to a very small slack).
    # Negative KL can happen due to numerical instability.
    with tf.control_dependencies([tf.debugging.assert_greater(kl_divergence, -1e-3, message="kl")]):
        kl_divergence = tf.identity(kl_divergence)

    elbo = reconstruction - kl_divergence
    avg_elbo = tf.reduce_mean(input_tensor=elbo)
    tf.summary.scalar("elbo", avg_elbo)
    loss = -avg_elbo

    # Perform variational inference by minimizing the -ELBO.
    global_step = tf.compat.v1.train.get_or_create_global_step()
    optimizer = tf.compat.v1.train.AdamOptimizer(params.learning_rate)

    # Add Horovod Distributed Optimizer.
    optimizer = hvd.DistributedOptimizer(optimizer)

    # This implements the "burn-in" for prior parameters
    # For the first prior_burn_in_steps steps they are fixed, and then trained jointly with the other parameters.
    grads_and_vars = optimizer.compute_gradients(loss)
    grads_and_vars_except_prior = [x for x in grads_and_vars if x[1] not in prior_variables]

    def train_op_except_prior() -> tf.Operation:
        """
        Apply gradients excluding prior.

        :return: Optimizer applying gradients
        """
        return optimizer.apply_gradients(grads_and_vars_except_prior, global_step=global_step)

    def train_op_all() -> tf.Operation:
        """
        Apply gradients with prior.

        :return: Optimizer applying gradients
        """
        return optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    train_op = tf.cond(
        pred=global_step < params.prior_burn_in_steps, true_fn=train_op_except_prior, false_fn=train_op_all
    )

    # The perplexity is an exponent of the average negative ELBO per word.
    words_per_document = tf.reduce_sum(input_tensor=features, axis=1)
    log_perplexity = -elbo / words_per_document
    tf.summary.scalar("perplexity", tf.exp(tf.reduce_mean(input_tensor=log_perplexity)))

    (log_perplexity_tensor, log_perplexity_update) = tf.compat.v1.metrics.mean(log_perplexity)
    perplexity_tensor = tf.exp(log_perplexity_tensor)

    # Obtain the topics summary.
    # Implemented as a py_func for simplicity.
    topics = tf.py_function(
        func=functools.partial(get_topics_strings, vocabulary=params.vocabulary),
        inp=[topics_words, alpha],
        Tout=tf.string,
    )

    tf.compat.v1.summary.text("topics", topics)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            "elbo": tf.compat.v1.metrics.mean(elbo),
            "reconstruction": tf.compat.v1.metrics.mean(reconstruction),
            "kl": tf.compat.v1.metrics.mean(kl_divergence),
            "perplexity": (perplexity_tensor, log_perplexity_update),
            "topics": (topics, tf.no_op()),
        },
    )


def get_topics_strings(topics_words: tf.Tensor, alpha: tf.Tensor, vocabulary: Dict[int, str]) -> np.array:
    """
    Return the summary of the learned topics.

    Show the top ten words matched with the top twenty topics.

    :param topics_words: KxV tensor with topics as rows and words as columns.
    :param alpha: 1xK tensor of prior Dirichlet concentrations for the topics.
    :param vocabulary: A mapping of word's integer index to the corresponding string.
    :return: A np.array with strings.
    """
    alpha = np.squeeze(alpha, axis=0)

    # Use a stable sorting algorithm so that when alpha is fixed we always get the same topics.
    highest_weight_topics = np.argsort(-alpha, kind="mergesort")
    top_words = np.argsort(-topics_words, axis=1)

    # Use string formatting to print a table of results.
    topics: List[str] = [f"{'index':<8} {'name':<20} {'alpha':<8} {'words':<120}"]

    for topic_index in highest_weight_topics[:20]:
        topic_name: str = vocabulary[top_words[topic_index][0]]
        topic_alpha: str = f"{round(alpha[topic_index], ndigits=4):.2f}"
        topic_words: str = " ".join([vocabulary[word] for word in top_words[topic_index, :10]])

        topics.append(f"{topic_index:<8} {topic_name:<20} {topic_alpha:<8} {topic_words:<120}")

    return np.array(topics)


def parse_dataset(path: str, num_words: int, shuffle_and_repeat: bool) -> tf.data.Dataset:
    """
    Return dataset as tf.data.Dataset.

    :param path: path to npz file
    :param num_words: number of words in dictionary
    :param shuffle_and_repeat: whether to shuffle and repeat the epoch
    :return: dataset as tf.data.Dataset
    """
    sparse_matrix = load_npz(path)
    num_documents = sparse_matrix.shape[0]
    dataset = tf.data.Dataset.range(num_documents)

    # For training, we shuffle each epoch and repeat the epochs.
    if shuffle_and_repeat:
        dataset = dataset.shuffle(num_documents).repeat()

    # Returns a single document as a dense TensorFlow tensor.
    # The dataset is stored as a sparse matrix outside of the graph.
    def get_row_python(index: int) -> np.ndarray:
        """
        Get dataset row as numpy array.

        :param index: row index
        :return: numpy nd array
        """
        return np.squeeze(np.array(sparse_matrix[index].todense()), axis=0)

    def get_row_py_func(index: int) -> tf.Tensor:
        """
        Set shape and type of dataset row.

        :param index: row index
        :return: dense Tensor of dataset row
        """
        py_func = tf.py_function(get_row_python, [index], tf.float32)
        py_func.set_shape((num_words,))
        return py_func

    dataset = dataset.map(get_row_py_func)
    return dataset


def build_input_fns(
    data_dir: str, batch_size: int
) -> Tuple[Callable[[], tf.data.Dataset], Callable[[], tf.data.Dataset], Dict[int, str]]:
    """
    Build iterators for train and evaluation data. Each object is represented as a bag-of-words vector.

    :param data_dir: Folder in which to store the data.
    :param batch_size: Batch size for both train and evaluation.
    :return:
        - A function that returns an iterator over the training data.
        - A function that returns an iterator over the evaluation data.
        - A mapping of word's integer index to the corresponding string.
    """
    with open(f"{data_dir}/dictionary_50000.pkl", "rb") as file:
        words_to_idx = pickle.load(file)  # nosec

    num_words = len(words_to_idx)
    vocabulary = {}
    for word, idx in words_to_idx.items():
        vocabulary[idx] = word

    def train_input_fn() -> tf.data.Dataset:
        """
        Load the train dataset.

        :return: tf data Dataset
        """
        train_dataset = parse_dataset(
            path=f"{data_dir}/train_data_50000.npz", num_words=num_words, shuffle_and_repeat=True
        )

        train_dataset = train_dataset.batch(batch_size).prefetch(batch_size)
        return train_dataset

    def eval_input_fn() -> tf.data.Dataset:
        """
        Load the test/eval dataset.

        :return: tf data Dataset
        """
        eval_dataset = parse_dataset(
            path=f"{data_dir}/test_data_50000.npz", num_words=num_words, shuffle_and_repeat=False
        )

        eval_dataset = eval_dataset.batch(batch_size)
        return eval_dataset

    return train_input_fn, eval_input_fn, vocabulary


def tensorflow_init() -> None:
    """
    Pipeline for topic modelling.

    :return: None
    """
    # Initialize Horovod
    hvd.init()

    parameters = Parameters()
    parameters.activation = getattr(tf.nn, parameters.activation)

    # Only save model in primary Horovod pod.
    if hvd.rank() == 0:
        tf.io.gfile.makedirs(parameters.model_dir)
    else:
        parameters.model_dir = None

    train_input_fn, eval_input_fn, vocabulary = build_input_fns(
        data_dir=PROCESSED_DIR, batch_size=parameters.batch_size
    )

    parameters.vocabulary = vocabulary

    # Pin GPU to be used to process local rank (one GPU per process)
    gpu_list = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpu_list:
        tf.config.experimental.set_visible_devices(gpu_list[hvd.local_rank()], "GPU")

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=parameters,
        config=tf.estimator.RunConfig(model_dir=parameters.model_dir, save_checkpoints_steps=parameters.viz_steps),
    )

    # BroadcastGlobalVariablesHook broadcasts initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when training with random weights or restored
    # from a checkpoint.
    broadcast_hook = hvd.BroadcastGlobalVariablesHook(0)

    for _ in range(parameters.max_steps // parameters.viz_steps):
        estimator.train(input_fn=train_input_fn, steps=parameters.viz_steps, hooks=[broadcast_hook])
        eval_results: Dict[str, Any] = estimator.evaluate(input_fn=eval_input_fn)  # type: ignore

        for key, value in eval_results.items():
            print(f"\n{key}")
            if key == "topics":
                for topic in value:
                    print(topic.decode())
            else:
                print(value)


def main() -> None:
    """
    Execute TensorFlow code across the Spark cluster.

    :return: None
    """
    spark.run(tensorflow_init)


if __name__ == "__main__":
    main()