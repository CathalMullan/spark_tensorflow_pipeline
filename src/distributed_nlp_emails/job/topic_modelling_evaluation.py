import functools
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.sparse import load_npz
from six.moves import cPickle as pickle

from distributed_nlp_emails.helpers.globals.directories import DATA_DIR, MODELS_DIR

tfd = tfp.distributions


class Config:
    layer_sizes: List[str] = [300, 300, 300]
    learning_rate: int = 3e-4
    max_steps: int = 180_000
    num_topics: int = 10
    activation: str = "relu"
    prior_initial_value: int = 0.7
    prior_burn_in_steps: int = 1_200
    model_dir: str = f"{MODELS_DIR}/topic_model_checkpoint"
    serving_dir: str = f"{MODELS_DIR}/topic_model_serving"
    viz_steps: int = 1_000
    vocabulary: str = None
    batch_size: int = 32


def clip_dirichlet_parameters(x):
    """
    Clips the Dirichlet parameters to the numerically stable KL region.
    """
    return tf.clip_by_value(x, 1e-3, 1e3)


def make_encoder(activation, num_topics, layer_sizes):
    """
    Create the encoder function.

    Args:
      activation: Activation function to use.
      num_topics: The number of topics.
      layer_sizes: The number of hidden units per layer in the encoder.

    Returns:
      encoder: A `callable` mapping a bag-of-words `Tensor` to a
        `tfd.Distribution` instance over topics.
    """
    encoder_net = tf.keras.Sequential()
    for num_hidden_units in layer_sizes:
        encoder_net.add(
            tf.keras.layers.Dense(num_hidden_units, activation=activation, kernel_initializer="glorot_normal")
        )

    encoder_net.add(tf.keras.layers.Dense(num_topics, activation=tf.nn.softplus, kernel_initializer="glorot_normal"))

    def encoder(bag_of_words):
        net = clip_dirichlet_parameters(encoder_net(bag_of_words))
        return tfd.Dirichlet(concentration=net, name="topics_posterior")

    return encoder


def make_decoder(num_topics, num_words):
    """
    Create the decoder function.

    Args:
      num_topics: The number of topics.
      num_words: The number of words.

    Returns:
      decoder: A `callable` mapping a `Tensor` of encodings to a
        `tfd.Distribution` instance over words.
    """
    glorot_normal_initializer = tf.initializers.glorot_normal()
    topics_words_logits = tf.Variable(
        name="topics_words_logits", initial_value=glorot_normal_initializer([num_topics, num_words])
    )

    topics_words = tf.nn.softmax(topics_words_logits, axis=-1)

    def decoder(topics):
        word_probs = tf.matmul(topics, topics_words)
        # The observations are bag of words and therefore not one-hot. However,
        # log_prob of OneHotCategorical computes the probability correctly in
        # this case.
        return tfd.OneHotCategorical(probs=word_probs, name="bag_of_words")

    return decoder, topics_words


def make_prior(num_topics, initial_value):
    """
    Create the prior distribution.

    Args:
      num_topics: Number of topics.
      initial_value: The starting value for the prior parameters.

    Returns:
      prior: A `callable` that returns a `tf.distribution.Distribution`
          instance, the prior distribution.
      prior_variables: A `list` of `Variable` objects, the trainable parameters
          of the prior.
    """
    softplus_inverse_initializer = tfp.math.softplus_inverse(
        tf.constant(value=initial_value, shape=[1, num_topics], dtype=tf.float32)
    )

    logit_concentration = tf.Variable(name="logit_concentration", initial_value=softplus_inverse_initializer)

    concentration = clip_dirichlet_parameters(tf.nn.softplus(logit_concentration))
    prior_variables = [logit_concentration]

    def prior():
        return tfd.Dirichlet(concentration=concentration, name="topics_prior")

    return prior, prior_variables


def model_fn(features, mode, params: Config):
    """
    Build the model function for use in an estimator.

    Arguments:
      features: The input features for the estimator.
      mode: Signifies whether it is train or test or predict.
      params: Some hyper-parameters as a dictionary.
    Returns:
      EstimatorSpec: A tf.estimator.EstimatorSpec instance.
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

    # Compute the KL-divergence between two Dirichlets analytically.
    # The sampled KL does not work well for "sparse" distributions
    kl = tfd.kl_divergence(topics_posterior, topics_prior)
    tf.summary.scalar("kl", tf.reduce_mean(input_tensor=kl))

    # Ensure that the KL is non-negative (up to a very small slack).
    # Negative KL can happen due to numerical instability.
    with tf.control_dependencies([tf.debugging.assert_greater(kl, -1e-3, message="kl")]):
        kl = tf.identity(kl)

    elbo = reconstruction - kl
    avg_elbo = tf.reduce_mean(input_tensor=elbo)
    tf.summary.scalar("elbo", avg_elbo)
    loss = -avg_elbo

    # Perform variational inference by minimizing the -ELBO.
    global_step = tf.compat.v1.train.get_or_create_global_step()
    optimizer = tf.compat.v1.train.AdamOptimizer(3e-4)

    # This implements the "burn-in" for prior parameters
    # For the first prior_burn_in_steps steps they are fixed, and then trained jointly with the other parameters.
    grads_and_vars = optimizer.compute_gradients(loss)
    grads_and_vars_except_prior = [x for x in grads_and_vars if x[1] not in prior_variables]

    def train_op_except_prior():
        return optimizer.apply_gradients(grads_and_vars_except_prior, global_step=global_step)

    def train_op_all():
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

    # Obtain the topics summary. Implemented as a py_func for simplicity.
    topics = tf.py_function(
        functools.partial(get_topics_strings, vocabulary=params.vocabulary), [topics_words, alpha], tf.string
    )

    tf.compat.v1.summary.text("topics", topics)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            "elbo": tf.compat.v1.metrics.mean(elbo),
            "reconstruction": tf.compat.v1.metrics.mean(reconstruction),
            "kl": tf.compat.v1.metrics.mean(kl),
            "perplexity": (perplexity_tensor, log_perplexity_update),
            "topics": (topics, tf.no_op()),
        },
    )


def get_topics_strings(topics_words, alpha, vocabulary):
    """
    Returns the summary of the learned topics.

    Arguments:
      topics_words: KxV tensor with topics as rows and words as columns.
      alpha: 1xK tensor of prior Dirichlet concentrations for the
          topics.
      vocabulary: A mapping of word's integer index to the corresponding string.
    Returns:
      summary: A np.array with strings.
    """
    alpha = np.squeeze(alpha, axis=0)
    # Use a stable sorting algorithm so that when alpha is fixed
    # we always get the same topics.
    highest_weight_topics = np.argsort(-alpha, kind="mergesort")
    top_words = np.argsort(-topics_words, axis=1)

    res = []
    for topic_idx in highest_weight_topics:
        line = [f"index = {topic_idx}, alpha = {alpha[topic_idx]:.2f},"]
        line += [vocabulary[word] for word in top_words[topic_idx, :10]]
        res.append(" ".join(line))

    return np.array(res)


def load_dataset(path, num_words, shuffle_and_repeat):
    """
    Return dataset as tf.data.Dataset.
    """
    sparse_matrix = load_npz(path)
    num_documents = sparse_matrix.shape[0]
    dataset = tf.data.Dataset.range(num_documents)

    # For training, we shuffle each epoch and repeat the epochs.
    if shuffle_and_repeat:
        dataset = dataset.shuffle(num_documents).repeat()

    # Returns a single document as a dense TensorFlow tensor.
    # The dataset is stored as a sparse matrix outside of the graph.
    def get_row_py_func(idx):
        def get_row_python(idx_py):
            return np.squeeze(np.array(sparse_matrix[idx_py].todense()), axis=0)

        py_func = tf.py_function(get_row_python, [idx], tf.float32)
        py_func.set_shape((num_words,))
        return py_func

    dataset = dataset.map(get_row_py_func)
    return dataset


def build_input_fns(data_dir, batch_size):
    """
    Builds iterators for train and evaluation data.

    Each object is represented as a bag-of-words vector.

    Arguments:
      data_dir: Folder in which to store the data.
      batch_size: Batch size for both train and evaluation.
    Returns:
      train_input_fn: A function that returns an iterator over the training data.
      eval_input_fn: A function that returns an iterator over the evaluation data.
      vocabulary: A mapping of word's integer index to the corresponding string.
    """
    with open(f"{data_dir}/dictionary_10000.pkl", "rb") as f:
        words_to_idx = pickle.load(f)

    num_words = len(words_to_idx)
    vocabulary = [None] * num_words
    for word, idx in words_to_idx.items():
        vocabulary[idx] = word

    # Build an iterator over training batches.
    def train_input_fn():
        train_dataset = load_dataset(f"{data_dir}/train_data_10000.npz", num_words, shuffle_and_repeat=True)
        train_dataset = train_dataset.batch(batch_size).prefetch(batch_size)
        return train_dataset

    # Build an iterator over the held-out set.
    def eval_input_fn():
        eval_dataset = load_dataset(f"{data_dir}/test_data_10000.npz", num_words, shuffle_and_repeat=False)
        eval_dataset = eval_dataset.batch(batch_size)
        return eval_dataset

    return train_input_fn, eval_input_fn, vocabulary


def main():
    global_config = Config()
    global_config.activation = getattr(tf.nn, global_config.activation)
    tf.io.gfile.makedirs(global_config.model_dir)

    train_input_fn, eval_input_fn, vocabulary = build_input_fns(
        data_dir=f"{DATA_DIR}/ignore", batch_size=global_config.batch_size
    )
    global_config.vocabulary = vocabulary

    estimator = tf.estimator.Estimator(
        model_fn,
        params=global_config,
        config=tf.estimator.RunConfig(
            model_dir=global_config.model_dir, save_checkpoints_steps=global_config.viz_steps
        ),
    )

    for _ in range(global_config.max_steps // global_config.viz_steps):
        estimator.train(train_input_fn, steps=global_config.viz_steps)
        eval_results = estimator.evaluate(eval_input_fn)

        for key, value in eval_results.items():
            print(f"{key}\n")
            if key == "topics":
                for topic in value:
                    print(topic)
                print("\n")
            else:
                print(f"{str(value)}")


if __name__ == "__main__":
    main()
