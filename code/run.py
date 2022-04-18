import random
import tensorflow as tf
import numpy as np

from code.algorithms.FairFed import FairFed
from code.datasets.distributions import get_x_dirichlet
from code.tensorflow.datasets_creator import get_tf_train_dataset, make_federated_data, get_tf_train_dataset_distributions


def run(dataset, num_rounds, num_runs, fl, alpha=None):
    n_clients = dataset.number_of_clients

    for run in range(1, num_runs + 1):
        print('RUN {:2d}'.format(run))
        seed = run * 10
        set_random_seeds(seed)
        x_train, y_train, x_test, y_test, x_val, y_val = dataset.train_val_test_split(seed)
        x_train_array, y_train_array = get_train_array_alpha(seed, alpha, dataset, x_train, y_train)

        clients_dataset, clients_dataset_x_y_label = get_clients_dataset_temp(alpha, x_train, y_train, x_train_array, y_train_array, n_clients)
        weights_local = dataset.get_weights_local(clients_dataset_x_y_label)
        federated_train_data = make_federated_data(
            dataset.sensitive_idx, clients_dataset, clients_dataset.client_ids[0:1], dataset.n_features,
            dataset.num_epochs, seed
        )
        fl.reset(federated_train_data, seed)

        for round_ in range(num_rounds):
            print('round {:2d}'.format(round_))
            clients_idx = generate_sample_clients_idx(dataset.num_clients_per_round, n_clients)
            clients = [clients_dataset.client_ids[i] for i in clients_idx]
            weights_global = dataset.get_weights_global([clients_dataset_x_y_label[i] for i in clients_idx], clients_idx)
            clients_dataset = get_clients_dataset(alpha, x_train, y_train, x_train_array, y_train_array, n_clients, weights_local, weights_global)
            federated_train_data = make_federated_data(
                dataset.sensitive_idx, clients_dataset, clients, dataset.n_features, dataset.num_epochs, seed
            )
            clients_data_size = [len(client_data) for client_data in [clients_dataset_x_y_label[i][0] for i in clients_idx]]
            if fl.name == FairFed.NAME:
                fl.iterate(dataset, federated_train_data, clients_dataset_x_y_label, None, x_test, y_test, clients_data_size, clients_idx)  # there is no validation set in the FairFed setup
            else:
                fl.iterate(dataset, federated_train_data, x_val, y_val, x_test, y_test, clients_data_size, None)
            
            
        fl.save_metrics_to_file(dataset.name, run, alpha)

def set_random_seeds(seed_value):
    #os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


def generate_sample_clients_idx(num_clients_per_round, number_of_clients):
    random_numbers = []

    while len(random_numbers) < num_clients_per_round:
        random_number = random.randint(0, number_of_clients - 1)
        if random_number not in random_numbers:
            random_numbers.append(random_number)

    return random_numbers


def get_clients_dataset(alpha, x_train, y_train, x_train_array, y_train_array, n_clients, weights_local, weights_global):
    if alpha:
        clients_dataset, _ = get_tf_train_dataset_distributions(x_train_array, y_train_array, n_clients, weights_local, weights_global)
    else:
        clients_dataset, _ = get_tf_train_dataset(x_train, y_train, n_clients, weights_local, weights_global)

    return clients_dataset


# temporary clients_dataset so that tensorflow knows shape
def get_clients_dataset_temp(alpha, x_train, y_train, x_train_array, y_train_array, n_clients):
    if alpha:
        _weights = [[0 for _ in range(len(x_train_array[i]))] for i in range(len(x_train_array))]
        clients_dataset, clients_dataset_x_y_label = get_tf_train_dataset_distributions(
            x_train_array, y_train_array, n_clients, _weights, _weights
        )
    else:
        _weights = [[0 for _ in range(len(x_train) // n_clients)] for _ in range(n_clients)]
        clients_dataset, clients_dataset_x_y_label = get_tf_train_dataset(
            x_train, y_train, n_clients, _weights, _weights
        )

    return clients_dataset, clients_dataset_x_y_label


def get_train_array_alpha(seed, alpha, dataset, x_train, y_train):
    if alpha:
        x_train_array, y_train_array = get_x_dirichlet(seed, alpha, dataset, x_train, y_train)
    else:
        x_train_array = None
        y_train_array = None

    return x_train_array, y_train_array
