import os
import random
import tensorflow as tf
import numpy as np

from code.algorithms.FairFate import FairFate
from code.algorithms.FedAvg import FedAvg
from code.algorithms.FedAvgLR import FedAvgLR
from code.algorithms.FedAvgGR import FedAvgGR
from code.algorithms.FedMom import FedMom
from code.datasets.distributions import get_x_dirichlet
from code.plots.pie_chart import create_stats_sensitive_distribution
from code.tensorflow.datasets_creator import get_tf_train_dataset, make_federated_data, get_tf_train_dataset_distributions


def run(dataset, num_rounds, num_runs, aggregation_metrics, alpha=None):
    n_clients = dataset.number_of_clients

    for run in range(1, num_runs + 1):
        print('RUN {:2d}'.format(run))
        seed = run * 10
        set_random_seeds(seed)
        df = dataset.preprocess()
        x_train, y_train, x_test, y_test, x_val, y_val = dataset.train_val_test_split(df, seed)
        n_features = len(x_train[0])
        if alpha:
            x_train_array, y_train_array = get_x_dirichlet(seed, alpha, dataset, x_train, y_train)
            _weights = [[0 for _ in range(len(x_train_array[i]))] for i in range(len(x_train_array))]
            clients_dataset, clients_dataset_x_y_label = get_tf_train_dataset_distributions(x_train_array, y_train_array, n_clients, _weights, _weights)
        else:
            _weights = [[0 for _ in range(len(x_train) // n_clients)] for _ in range(n_clients)]
            clients_dataset, clients_dataset_x_y_label = get_tf_train_dataset(x_train, y_train, n_clients, _weights, _weights)

        if alpha:
            x_ys = [[x_val, y_val, "VAL"], [x_test, y_test, "TEST"], *clients_dataset_x_y_label]
            create_stats_sensitive_distribution(x_ys, dataset, alpha, "./datasets/{}/run_{}".format(dataset.name, run))

        sensitive_idx = [df.columns.get_loc(s.name) for s in dataset.sensitive_attributes]
        federated_train_data = make_federated_data(sensitive_idx, clients_dataset, clients_dataset.client_ids[0:1], n_features, seed)
        weights_local = dataset.get_weights_local(clients_dataset_x_y_label)
        fls = create_fls(federated_train_data, n_features, dataset, aggregation_metrics)

        for round_ in range(num_rounds):
            print('round {:2d}'.format(round_))
            clients_idx = generate_sample_clients_idx(dataset.num_clients_per_round, n_clients)
            clients = [clients_dataset.client_ids[i] for i in clients_idx]
            weights_global = dataset.get_weights_global([clients_dataset_x_y_label[i] for i in clients_idx], clients_idx)
            if alpha:
                clients_dataset, _ = get_tf_train_dataset_distributions(x_train_array, y_train_array, n_clients, weights_global, weights_local)
            else:
                clients_dataset, _ = get_tf_train_dataset(x_train, y_train, n_clients, weights_global, weights_local)
            federated_train_data = make_federated_data(sensitive_idx, clients_dataset, clients, n_features, seed)
            clients_data_size = [len(client_data) for client_data in [clients_dataset_x_y_label[i][0] for i in clients_idx]]

            for fl in fls:
                fl.iterate(dataset, federated_train_data, n_features, x_val, y_val, x_test, y_test, clients_data_size)

        for fl in fls:
            fl.save_metrics_to_file(dataset.name, run, alpha)


def set_random_seeds(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


def create_fls(federated_train_data, n_features, dataset, aggregation_metrics):
    fls = []
    """
    fls = [
        FedAvg(federated_train_data, n_features),
        FedAvgGR(federated_train_data, n_features),
        FedAvgLR(federated_train_data, n_features)
    ]

    # FedMom
    for beta in [0.7, 0.8, 0.9, 0.99]:
        fls.append(FedMom(federated_train_data, n_features, dataset, beta=beta))
    """
    for metric in aggregation_metrics:
        metric.reset()

    for beta in [0.7, 0.8, 0.9, 0.99]:
        for lambda_exponential in [0.04, 0.045, 0.05]:
            fls.append(FairFate(federated_train_data, n_features, dataset, aggregation_metrics, lambda_exponential=lambda_exponential, beta=beta))

    return fls


def generate_sample_clients_idx(num_clients_per_round, number_of_clients):
    random_numbers = []

    while len(random_numbers) < num_clients_per_round:
        random_number = random.randint(0, number_of_clients - 1)

        if random_number not in random_numbers:
            random_numbers.append(random_number)

    return random_numbers
