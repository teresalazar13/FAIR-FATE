from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.FLClientSide import FederatedLearningClientSide
from code.tensorflow.models import get_model

import numpy as np


class FedAvg(FederatedLearningAlgorithm):
    def __init__(self, dataset):
        name = "fedavg"
        super().__init__(name)

        self.dataset = dataset

    def reset(self, federated_train_data, seed):
        algorithm = FederatedLearningClientSide(
            False, federated_train_data, self.dataset.n_features, self.dataset.learning_rate, seed
        )
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

    def update(self, weights, unused_x_val, unused_y_val, clients_data_size, _):
        return fed_avg_update(weights, self.dataset.n_features, clients_data_size)


def fed_avg_update(weights, n_features, clients_data_size):
    model = get_model(n_features)
    new_state = []
    n_layers = len(weights[0])
    sum_clients_data_size = sum(clients_data_size)

    for l in range(n_layers):
        sum_ = np.array(weights[0][l] * clients_data_size[0] / sum_clients_data_size)
        for c in range(1, len(weights)):
            sum_ = np.add(sum_, weights[c][l] * clients_data_size[c] / sum_clients_data_size)
        new_state.append(sum_)

    model.set_weights(new_state)

    return new_state, model
