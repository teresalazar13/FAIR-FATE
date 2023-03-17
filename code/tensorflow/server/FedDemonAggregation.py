from copy import deepcopy
import numpy as np
import math

from code.tensorflow.models import get_model


class FedDemonAggregation:

    def __init__(self, state, dataset, beta=0.9):
        self.actual_state = deepcopy(state)  # weights
        self.dataset = dataset
        self.momentum = self._get_model_shape()  # copy shape of state with zeros
        self.beta_init = beta
        self.iteration = 1  # round_num
        self.beta = self.get_beta()

    def get_beta(self):
        beta = self.beta_init * (1 - self.iteration/100) / ((1 - self.beta_init) + self.beta_init*(1 - self.iteration/100))

        return beta

    def _get_model_shape(self):
        momentum = []
        for layer in self.actual_state:
            momentum.append(np.zeros_like(layer))
        return momentum

    # Calculate a1, a2, a3 (w0-W, w1-W, w3-W)
    def _calculate_local_update(self, clients_weights):
        state_update = []
        n_layers = len(clients_weights[0])

        for c in range(self.dataset.num_clients_per_round):
            layers = []
            for layer in range(n_layers):
                layers.append(clients_weights[c][layer] - self.actual_state[layer])
            state_update.append(layers)

        return state_update

    def calculate_momentum_update_layer(self, layer, state_update_clients, clients_data_size):
        new_state_layer = np.zeros_like(state_update_clients[0][layer])
        sum_clients_data_size = sum(clients_data_size)

        for c in range(self.dataset.num_clients_per_round):
            new_state_layer = np.add(
                new_state_layer,
                np.multiply(state_update_clients[c][layer], clients_data_size[c] / sum_clients_data_size)
            )

        return calculate_momentum(self.momentum[layer], self.beta, new_state_layer)

    def update_model(self, clients_weights, n_features, clients_data_size):
        model = get_model(n_features)
        state_update_clients = self._calculate_local_update(clients_weights)  # alphas

        new_state_with_momentum = []
        for layer in range(len(clients_weights[0])):
            update_m = self.calculate_momentum_update_layer(layer, state_update_clients, clients_data_size)
            self.momentum[layer] = deepcopy(update_m)
            bias_correction = update_m / (1.0 - math.pow(self.beta, self.iteration))
            new_state_with_momentum.append(self.actual_state[layer] + bias_correction)

        self.actual_state = deepcopy(new_state_with_momentum)
        model.set_weights(self.actual_state)
        self.iteration += 1
        self.beta = self.get_beta()

        return new_state_with_momentum, model


def calculate_momentum(momentum_l, beta, new_state_layer):
    return np.array(beta * momentum_l) + np.array((1.0 - beta) * new_state_layer)
