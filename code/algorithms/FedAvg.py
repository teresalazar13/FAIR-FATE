from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.FLClientSide import FederatedLearningClientSide
from code.tensorflow.models import get_model

import numpy as np


class FedAvg(FederatedLearningAlgorithm):
    def __init__(self, federated_train_data, x_train):
        name = "fedavg"
        algorithm = FederatedLearningClientSide(0, federated_train_data, x_train[0])
        state = algorithm.initialize()
        super().__init__(name, algorithm, state)

    def update(self, weights, x_train, unused_x_val, unused_y_val):
        return fed_avg_update(weights, x_train)


def fed_avg_update(weights, x_train):
    model = get_model(x_train)
    new_state = []
    n_layers = len(weights[0])

    for l in range(n_layers):
        sum_ = np.array(weights[0][l])
        for c in range(1, len(weights)):
            sum_ = np.add(sum_, weights[c][l])
        sum_ = sum_ / len(weights)
        new_state.append(sum_)

    model.set_weights(new_state)

    return new_state, model
