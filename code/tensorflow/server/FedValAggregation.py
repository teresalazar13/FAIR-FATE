from copy import deepcopy
import numpy as np

from code.metrics.retriever import get_fairness
from code.tensorflow.models import get_model


class FedValAggregation:

    def __init__(self, state, dataset, aggregation_metrics):
        self.actual_state = deepcopy(state)  # weights
        self.dataset = dataset
        self.aggregation_metrics = aggregation_metrics

    def calculate_fairness_clients(self, clients_weights, model, x_val, y_val):
        fairness_clients = []

        for client in clients_weights:
            model.set_weights(client)
            y_pred = model.predict(x_val)
            fairness = get_fairness(self.dataset, x_val, y_pred, y_val, self.aggregation_metrics, debug=False)
            fairness_clients.append(fairness)

        return fairness_clients

    # calculate the new model
    def calculate_fair_update_layer(self, layer, state_update_clients, fairness_clients):
        fairness_sum_fair_clients = 0
        new_state_layer = np.zeros_like(state_update_clients[0][layer])

        for c in range(self.dataset.num_clients_per_round):
            new_state_layer = np.add(
                new_state_layer,
                np.multiply(state_update_clients[c][layer], fairness_clients[c])
            )
            fairness_sum_fair_clients += fairness_clients[c]

        if fairness_sum_fair_clients == 0:
            return self.actual_state[layer]
        else:
            return new_state_layer / fairness_sum_fair_clients

    def update_model(self, clients_weights, n_features, x_val, y_val):
        model = get_model(n_features)

        fairness_clients = self.calculate_fairness_clients(clients_weights, model, x_val, y_val)
        fairness_clients = avg_fairness(fairness_clients)

        new_state = []
        for layer in range(len(clients_weights[0])):
            update_fair = self.calculate_fair_update_layer(
                layer, clients_weights, fairness_clients
            )
            new_state.append(update_fair)

        self.actual_state = deepcopy(new_state)
        model.set_weights(self.actual_state)

        return new_state, model


def avg_fairness(fairness_clients):
    fairness_clients_avg = [0.001 for _ in fairness_clients]
    #weights = [0.999, 0.001]

    for j in range(len(fairness_clients[0])):  # for each metric
        for i in range(len(fairness_clients)):  # for each client
            fairness_clients_avg[i] += fairness_clients[i][j] #* weights[j]

    return fairness_clients_avg
