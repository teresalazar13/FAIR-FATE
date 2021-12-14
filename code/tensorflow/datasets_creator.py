import numpy as np
import collections
import tensorflow as tf
import tensorflow_federated as tff


def get_tf_train_dataset(x_train, y_train, number_of_clients, weights_global, weights_local):
    n_instances_client = int(np.floor(len(x_train) / number_of_clients))
    client_train_dataset_ = collections.OrderedDict()
    client_train_dataset_x_y_label = []

    for i in range(1, number_of_clients + 1):
        client_name = "client_" + str(i)
        start = n_instances_client * (i - 1)
        end = n_instances_client * i

        data = collections.OrderedDict((
            ('y', y_train[start:end]),
            ('x', x_train[start:end]),
            ('reweighting_weights_global', tf.cast(weights_global[i - 1], tf.float32)),
            ('reweighting_weights_local', tf.cast(weights_local[i - 1], tf.float32))
        ))
        client_train_dataset_[client_name] = data
        client_train_dataset_x_y_label.append(
            [x_train[start:end], y_train[start:end].reshape(len(y_train[start:end]), 1), client_name])

    return tff.simulation.datasets.TestClientData(client_train_dataset_), client_train_dataset_x_y_label
    # TODO - TestClientData? instead of ClientData


def get_client_tf_dataset(sensitive_idx, dataset_for_client, x_train, seed):
    num_epochs = 10  # local epochs
    batch_size = 10
    shuffle_buffer = 100
    prefetch_buffer = 10
    sensitive_columns_indexes_tf = tf.cast(sensitive_idx, tf.int64)

    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""

        return collections.OrderedDict(
            x=tf.reshape(element['x'], [-1, len(x_train[0])]),
            y=tf.reshape(element['y'], [-1, 1]),
            reweighting_weights_global=tf.reshape(element['reweighting_weights_global'], [-1, 1]),
            reweighting_weights_local=tf.reshape(element['reweighting_weights_local'], [-1, 1]),
            weights=tf.reshape(tf.gather(element['x'], sensitive_columns_indexes_tf, axis=1), [-1, len(sensitive_idx)])
        )

    return dataset_for_client.repeat(num_epochs).shuffle(shuffle_buffer, seed=seed).batch(batch_size).map(batch_format_fn).prefetch(prefetch_buffer)


def make_federated_data(sensitive_idx, client_data, clients, x_train, seed):
    federated_data = []

    for i in range(len(clients)):
        federated_data.append(
            get_client_tf_dataset(sensitive_idx, client_data.create_tf_dataset_for_client(clients[i]), x_train, seed))

    return federated_data
