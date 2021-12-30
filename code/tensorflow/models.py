import collections
import tensorflow as tf
import tensorflow_federated as tff


def model_fn(federated_train_data, n_features, seed):
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.

    initializer = tf.keras.initializers.RandomNormal(seed=seed)
    keras_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(n_features,)),
        tf.keras.layers.Dense(10, activation='tanh', kernel_initializer=initializer),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer),
    ])

    element_spec = collections.OrderedDict(list(federated_train_data[0].element_spec.items())[0:2])

    return tff.learning.from_keras_model(
        keras_model,
        input_spec=element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])


def get_model(n_features):
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(n_features,)),
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
