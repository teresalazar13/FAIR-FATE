import os
import random
import tensorflow as tf
import numpy as np

from code.algorithms.FairFate import FairFate
from code.algorithms.FedAvg import FedAvg
from code.algorithms.FedAvgAccRatio import FedAvgAccRatio
from code.algorithms.FedAvgLR import FedAvgLR
from code.algorithms.FedAvgGR import FedAvgGR
from code.algorithms.FedMom import FedMom
from code.plots.pie_chart import create_stats_sensitive_distribution
from code.tensorflow.datasets_creator import get_tf_train_dataset, make_federated_data


def run(dataset, num_rounds, num_runs):
    for run_num in range(1, num_runs + 1):
        print('RUN {:2d}'.format(run_num))

        seed = run_num * 10
        set_random_seeds(seed)

        df = dataset.preprocess()
        sensitive_columns_indexes = [df.columns.get_loc(s.name) for s in dataset.sensitive_attributes]
        x_train, y_train, x_test, y_test, x_val, y_val = dataset.train_val_test_split(df, seed)

        reweighting_weights_global = [[0 for _ in range(len(x_train) // dataset.number_of_clients)] for _ in
                                      range(dataset.number_of_clients)]
        reweighting_weights_local = [[0 for _ in range(len(x_train) // dataset.number_of_clients)] for _ in
                                     range(dataset.number_of_clients)]
        client_train_dataset, client_train_dataset_x_y_label = get_tf_train_dataset(x_train, y_train,
                                                                                    dataset.number_of_clients,
                                                                                    reweighting_weights_global,
                                                                                    reweighting_weights_local)

        """
        create_stats_sensitive_distribution(
            [[x_val, y_val, "VAL"], [x_test, y_test, "TEST"], *client_train_dataset_x_y_label], dataset,
            "./datasets/{}/run_{}".format(dataset.name, run_num))"""
        federated_train_data = make_federated_data(sensitive_columns_indexes, client_train_dataset,
                                                   client_train_dataset.client_ids[0:1], x_train, seed)
        reweighting_weights_local = dataset.get_reweigthing_weights_local(client_train_dataset_x_y_label)

        fls = [
            FedAvg(federated_train_data, x_train),
            FedAvgGR(federated_train_data, x_train),
            FedAvgLR(federated_train_data, x_train),
            FedAvgAccRatio(federated_train_data, x_train, dataset)
        ]

        # FedMom
        for beta in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
            fls.append(FedMom(federated_train_data, x_train, dataset, beta=beta))
        # FAIR-FATE
        for beta in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
            for lambda_ in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
                fls.append(FairFate(federated_train_data, x_train, dataset, lambda_=lambda_, beta=beta))

        for round_num in range(num_rounds):
            print('round {:2d}'.format(round_num))
            sample_clients_idx = generate_sample_clients_idx(dataset.num_clients_per_round, dataset.number_of_clients)
            sample_clients = [client_train_dataset.client_ids[i] for i in sample_clients_idx]
            sample_clients_x_y_label = [client_train_dataset_x_y_label[i] for i in sample_clients_idx]
            reweighting_weights_global = dataset.get_reweigthing_weights_global(sample_clients_x_y_label,
                                                                                sample_clients_idx)

            client_train_dataset, client_train_dataset_x_y_label = get_tf_train_dataset(x_train, y_train,
                                                                                        dataset.number_of_clients,
                                                                                        reweighting_weights_global,
                                                                                        reweighting_weights_local)
            federated_train_data = make_federated_data(sensitive_columns_indexes, client_train_dataset, sample_clients,
                                                       x_train, seed)

            for i in range(len(fls)):
                fls[i].iterate(dataset, federated_train_data, x_train, x_val, y_val, x_test, y_test)

        for fl in fls:
            fl.save_metrics_to_file(dataset.name, run_num)


def set_random_seeds(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
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
