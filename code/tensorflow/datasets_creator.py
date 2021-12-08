import numpy as np
import collections
import tensorflow as tf
import tensorflow_federated as tff


def get_tf_train_dataset(x_train, y_train, number_of_clients, reweighting_weights_global, reweighting_weights_local):
    image_per_set = int(np.floor(len(x_train) / number_of_clients))
    client_train_dataset_ = collections.OrderedDict()
    client_train_dataset_x_y_label = []

    for i in range(1, number_of_clients + 1):
        client_name = "client_" + str(i)
        start = image_per_set * (i - 1)
        end = image_per_set * i

        # print(f"Adding data from {start} to {end} for client : {client_name}")
        data = collections.OrderedDict((
            ('y', y_train[start:end]),
            ('x', x_train[start:end]),
            ('reweighting_weights_global', tf.cast(reweighting_weights_global[i - 1], tf.float32)),
            ('reweighting_weights_local', tf.cast(reweighting_weights_local[i - 1], tf.float32))
        ))
        client_train_dataset_[client_name] = data
        client_train_dataset_x_y_label.append(
            [x_train[start:end], y_train[start:end].reshape(len(y_train[start:end]), 1), client_name])

    return tff.simulation.datasets.TestClientData(client_train_dataset_), client_train_dataset_x_y_label
    # TODO - TestClientData? instead of ClientData


def get_client_tf_dataset(sensitive_columns_indexes, dataset_for_client, x_train, seed):
    num_epochs = 10  # local epochs
    batch_size = 10
    shuffle_buffer = 100
    prefetch_buffer = 10
    sensitive_columns_indexes_tf = tf.cast(sensitive_columns_indexes, tf.int64)

    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""

        return collections.OrderedDict(
            x=tf.reshape(element['x'], [-1, len(x_train[0])]),
            y=tf.reshape(element['y'], [-1, 1]),
            reweighting_weights_global=tf.reshape(element['reweighting_weights_global'], [-1, 1]),
            reweighting_weights_local=tf.reshape(element['reweighting_weights_local'], [-1, 1]),
            weights=tf.reshape(tf.gather(element['x'], sensitive_columns_indexes_tf, axis=1),
                               [-1, len(sensitive_columns_indexes)])
        )

    return dataset_for_client.repeat(num_epochs).shuffle(shuffle_buffer, seed=seed).batch(batch_size).map(
        batch_format_fn).prefetch(prefetch_buffer)


def make_federated_data(sensitive_columns_indexes, client_data, sample_clients, x_train, seed):
    federated_data = []

    for i in range(len(sample_clients)):
        federated_data.append(
            get_client_tf_dataset(sensitive_columns_indexes,
                                  client_data.create_tf_dataset_for_client(sample_clients[i]), x_train, seed)
        )

    return federated_data
