from copy import deepcopy
import numpy as np
import math

from code.metrics.retriever import get_fairness
from code.tensorflow.models import get_model


class FederatedFairMomentum:

    def __init__(self, state, dataset, beta=0.9, aggregation_metric="EO"):
        self.actual_state = deepcopy(state)  # weights
        self.dataset = dataset
        self.momentum = self._get_model_shape()  # copy shape of state with zeros
        self.lambda_ = .5
        self.beta = beta
        self.aggregation_metric = aggregation_metric
        self.iteration = 1  # round_num

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

    def calculate_fairness_server(self, model, x_val, y_val):
        model.set_weights(self.actual_state)
        y_pred = model.predict(x_val)

        return get_fairness(self.aggregation_metric, self.dataset, x_val, y_pred, y_val) + 1

    def calculate_fairness_clients(self, clients_weights, model, x_val, y_val):
        weights_client = []

        for client in clients_weights:
            model.set_weights(client)
            y_pred = model.predict(x_val)
            eo = get_fairness(self.aggregation_metric, self.dataset, x_val, y_pred, y_val)
            weights_client.append(eo + 1)

        return weights_client

    # calculate the new model (new_state), aN = (a1*w1+a2*w2+...+an*wn) / N
    def calculate_momentum_update_layer(self, layer, state_update_clients):
        new_state_layer = np.zeros_like(state_update_clients[0][layer])

        for c in range(self.dataset.num_clients_per_round):
            new_state_layer = np.add(
                new_state_layer,
                np.multiply(state_update_clients[c][layer], 1 / self.dataset.num_clients_per_round)
            )

        mom_bias_correction = self.momentum[layer] / (1.0 - math.pow(self.beta, self.iteration))
        update_m = calculate_momentum(mom_bias_correction, self.lambda_, new_state_layer)

        return update_m

    # calculate the new model (new_state for momentum), aF
    def calculate_momentum_fair_update_layer(self, layer, state_update_clients, fairness_clients, server_eo):
        fairness_sum_fair_clients = 0
        new_state_layer = np.zeros_like(state_update_clients[0][layer])

        for c in range(self.dataset.num_clients_per_round):
            if fairness_clients[c] > server_eo:
                new_state_layer = np.add(
                    new_state_layer,
                    np.multiply(state_update_clients[c][layer], fairness_clients[c])
                )
                fairness_sum_fair_clients += fairness_clients[c]

        if fairness_sum_fair_clients == 0:
            new_state_layer = np.zeros_like(new_state_layer)
        else:
            new_state_layer = new_state_layer / fairness_sum_fair_clients

        return calculate_momentum(self.momentum[layer], self.beta, new_state_layer)

    def update_model(self, clients_weights, x_train, x_val, y_val):
        model = get_model(x_train)
        state_update_clients = self._calculate_local_update(clients_weights)  # alphas

        fairness_clients = self.calculate_fairness_clients(clients_weights, model, x_val, y_val)
        fairness_server = self.calculate_fairness_server(model, x_val, y_val)

        new_state_with_momentum = []
        for layer in range(len(clients_weights[0])):  # for each layer

            update_m_fair = self.calculate_momentum_fair_update_layer(layer, state_update_clients, fairness_clients,
                                                                      fairness_server)
            self.momentum[layer] = deepcopy(update_m_fair)

            update_m = self.calculate_momentum_update_layer(layer, state_update_clients)
            new_state_with_momentum.append(self.actual_state[layer] + update_m)

        self.actual_state = deepcopy(new_state_with_momentum)
        model.set_weights(self.actual_state)

        return new_state_with_momentum, model


def calculate_momentum(momentum_l, beta, new_state_layer):
    return np.array(beta * momentum_l) + (1 - beta) * new_state_layer
