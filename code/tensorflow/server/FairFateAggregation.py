from copy import deepcopy
import numpy as np
import math

from code.metrics.retriever import get_fairness
from code.tensorflow.models import get_model


class FairFateAggregation:

    def __init__(self, state, dataset, aggregation_metrics, MAX=1000, beta0=0.9, l0=0.1, rho=0.05):
        self.actual_state = deepcopy(state)  # weights
        self.dataset = dataset
        self.aggregation_metrics = aggregation_metrics
        self.momentum = self._get_model_shape()  # copy shape of state with zeros
        self.iteration = 1  # round_num
        self.l0 = l0
        self.MAX = MAX
        self.lambda_ = self.get_lambda()
        self.beta_init = beta0
        self.beta = self.get_beta()
        self.rho = rho
        self.epsilon = 0.0001

    def get_beta(self):  # TODO - replace 100 with num_rounds
        beta = self.beta_init * (1 - self.iteration/100) / ((1 - self.beta_init) + self.beta_init*(1 - self.iteration/100))

        return beta

    def get_lambda(self):
        lambda_ = self.l0 * ((1 + self.rho) ** self.iteration)
        if lambda_ > self.MAX:
            lambda_ = self.MAX

        return lambda_

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

        return get_fairness(self.dataset, x_val, y_pred, y_val, self.aggregation_metrics, debug=False)

    def calculate_fairness_clients(self, clients_weights, model, x_val, y_val):
        fairness_clients = []

        for client in clients_weights:
            model.set_weights(client)
            y_pred = model.predict(x_val)
            fairness = get_fairness(self.dataset, x_val, y_pred, y_val, self.aggregation_metrics, debug=False)
            fairness_clients.append(fairness)

        return fairness_clients

    # calculate lambda * v_{t+1} + (1-lambda) alpha_N, alpha_N = (a1*w1+a2*w2+...+an*wn) / N
    def calculate_model_update_layer(self, layer, state_update_clients, clients_data_size):
        alpha_N = np.zeros_like(state_update_clients[0][layer])
        sum_clients_data_size = sum(clients_data_size)

        for c in range(self.dataset.num_clients_per_round):
            alpha_N = np.add(
                alpha_N,
                np.multiply(state_update_clients[c][layer], clients_data_size[c] / sum_clients_data_size)
            )

        mom_bias_correction = self.momentum[layer] / (1.0 - math.pow(self.beta, self.iteration))
        update_m = np.array(self.lambda_ * mom_bias_correction) + (1 - self.lambda_) * alpha_N

        return update_m

    # calculate v_{t+1} of layer
    def calculate_momentum_update_layer(self, layer, state_update_clients, fairness_clients, fairness_server):
        fairness_sum_fair_clients = 0
        alpha_F = np.zeros_like(state_update_clients[0][layer])

        for c in range(self.dataset.num_clients_per_round):
            if fairness_clients[c] > fairness_server:
                alpha_F = np.add(
                    alpha_F,
                    np.multiply(state_update_clients[c][layer], fairness_clients[c])
                )
                fairness_sum_fair_clients += fairness_clients[c]

        if fairness_sum_fair_clients == 0:
            alpha_F = np.zeros_like(self.momentum[layer])
        else:
            alpha_F = alpha_F / fairness_sum_fair_clients

        if self.beta:
            np.array(self.beta * self.momentum[layer]) + (1 - self.beta) * alpha_F
        else:  # for ablation study with no momentum
            return alpha_F

    def update_model(self, clients_weights, n_features, x_val, y_val, clients_data_size):
        model = get_model(n_features)
        state_update_clients = self._calculate_local_update(clients_weights)  # alphas

        fairness_server = self.calculate_fairness_server(model, x_val, y_val)
        fairness_clients = self.calculate_fairness_clients(clients_weights, model, x_val, y_val)
        fairness_clients, fairness_server = normalize_avg_fairness(fairness_clients, fairness_server)

        new_state_with_momentum = []
        for layer in range(len(clients_weights[0])):
            momentum_update_layer = self.calculate_momentum_update_layer(
                layer, state_update_clients, fairness_clients, fairness_server
            )
            self.momentum[layer] = deepcopy(momentum_update_layer)

            update_m = self.calculate_model_update_layer(layer, state_update_clients, clients_data_size)
            new_state_with_momentum.append(self.actual_state[layer] + update_m)

        self.actual_state = deepcopy(new_state_with_momentum)
        model.set_weights(self.actual_state)
        self.iteration += 1
        self.lambda_ = self.get_lambda()
        self.beta = self.get_beta()

        return new_state_with_momentum, model


def normalize_avg_fairness(fairness_clients, fairness_server):
    fairness_clients_avg = [0 for _ in fairness_clients]
    fairness_server_avg = 0

    for j in range(len(fairness_clients[0])):  # for each metric
        f_server = fairness_server[j]
        fs = [f_server]
        for i in range(len(fairness_clients)):  # for each client
            f_client = fairness_clients[i][j]
            fs.append(f_client)
        min_fs = min(fs)
        max_fs = max(fs)

        for i in range(len(fairness_clients)):  # for each client
            fairness_clients_avg[i] += normalize(fairness_clients[i][j], min_fs, max_fs)
            #fairness_clients_avg[i] += fairness_clients[i][j]
        fairness_server_avg += normalize(f_server, min_fs, max_fs)
        #fairness_server_avg += f_server

    return fairness_clients_avg, fairness_server_avg


def normalize(data, min_, max_):
    if max_ - min_ == 0:
        return 0
    return (data - min_) / (max_ - min_)
