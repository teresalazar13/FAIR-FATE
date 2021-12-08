import collections
import tensorflow as tf
import tensorflow_federated as tff


def model_fn(federated_train_data, input_layer_shape):
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(len(input_layer_shape),)),
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    element_spec = collections.OrderedDict(list(federated_train_data[0].element_spec.items())[0:2])

    return tff.learning.from_keras_model(
        keras_model,
        input_spec=element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])


def get_model(x_train):
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(len(x_train[0]),)),
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
