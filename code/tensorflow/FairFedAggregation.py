import math
from copy import deepcopy
import numpy as np

from code.metrics.retriever import get_fairness, create_dataframe_for_eval
from code.tensorflow.models import get_model


class FairFedAggregation:

    def __init__(self, state, dataset, aggregation_metrics, beta):
        self.actual_state = deepcopy(state)  # weights
        self.dataset = dataset
        self.aggregation_metrics = aggregation_metrics
        self.beta = beta

    def calculate_fairness_weights(self, model, clients_dataset_x_y_label, clients_idx):
        n = []
        denominator_unpriv = []
        denominator_priv = []
        numerator_unpriv = []
        numerator_priv = []
        local_fairness_clients = []

        for idx in clients_idx:
            n.append(len(clients_dataset_x_y_label[idx][0]))
            y_pred = model.predict(clients_dataset_x_y_label[idx][0])
            df = create_dataframe_for_eval(self.dataset.all_columns, clients_dataset_x_y_label[idx][0], y_pred, clients_dataset_x_y_label[idx][1])
            sensitive_attribute = self.dataset.sensitive_attributes[0].name
            metric = self.aggregation_metrics[0].name

            [
                denominator_unpriv_client_array,
                denominator_priv_client_array,
                numerator_unpriv_client_array,
                numerator_priv_client_array
            ] = calculate_global_fairness_probs(metric, df, sensitive_attribute)
            denominator_unpriv.append(denominator_unpriv_client_array)
            denominator_priv.append(denominator_priv_client_array)
            numerator_unpriv.append(numerator_unpriv_client_array)
            numerator_priv.append(numerator_priv_client_array)

            fairness = get_fairness(self.dataset, clients_dataset_x_y_label[idx][0], y_pred, clients_dataset_x_y_label[idx][1], self.aggregation_metrics, debug=False)
            local_fairness_clients.append(fairness)

        global_fairness = calculate_global_fairness(n, denominator_unpriv, denominator_priv, numerator_unpriv, numerator_priv)
        if global_fairness == -1:
            return [1 for _ in range(len(local_fairness_clients))]

        weight_clients = []
        for i in range(len(local_fairness_clients)):
            weight = math.exp(-self.beta * abs(local_fairness_clients[i][0] - global_fairness)) * n[i] / sum(n)
            weight_clients.append(weight)

        if sum(weight_clients) == 0:
            return [1 for _ in range(len(local_fairness_clients))]

        total = sum(weight_clients)
        for i in range(len(weight_clients)):
            weight_clients[i] = weight_clients[i] / total

        return weight_clients

    def update_model(self, clients_weights, n_features, clients_dataset_x_y_label, clients_idx):
        model = get_model(n_features)
        fairness_weights = self.calculate_fairness_weights(model, clients_dataset_x_y_label, clients_idx)

        new_state = []
        for layer in range(len(clients_weights[0])):
            update_fair = self.calculate_fair_update_layer(layer, clients_weights, fairness_weights)
            new_state.append(update_fair)

        self.actual_state = deepcopy(new_state)
        model.set_weights(self.actual_state)

        return new_state, model

    # calculate the new model
    def calculate_fair_update_layer(self, layer, state_update_clients, fairness_weights):
        new_state_layer = np.zeros_like(state_update_clients[0][layer])

        if sum(fairness_weights) == 0:
            return self.actual_state[layer]

        for c in range(self.dataset.num_clients_per_round):
            new_state_layer = np.add(
                new_state_layer,
                np.multiply(state_update_clients[c][layer], fairness_weights[c])
            )
        return new_state_layer


def calculate_global_fairness_probs(metric, df, sensitive_attribute):
    if metric == "EQO" or metric == "TPR":
        denominator_unpriv_client = len(df[(df[sensitive_attribute] == 0) & (df["y"] == 1)])
        denominator_priv_client = len(df[(df[sensitive_attribute] == 1) & (df["y"] == 1)])
        numerator_unpriv_client = len(df[(df[sensitive_attribute] == 0) & (df["y"] == 1) & (df["y_pred"] == 1)])
        numerator_priv_client = len(df[(df[sensitive_attribute] == 1) & (df["y"] == 1) & (df["y_pred"] == 1)])

        if metric == "TPR":
            denominator_unpriv_client_array = [denominator_unpriv_client]
            denominator_priv_client_array = [denominator_priv_client]
            numerator_unpriv_client_array = [numerator_unpriv_client]
            numerator_priv_client_array = [numerator_priv_client]
        else:
            denominator_unpriv_client_2 = len(df[(df[sensitive_attribute] == 0) & (df["y"] == 0)])
            denominator_priv_client_2 = len(df[(df[sensitive_attribute] == 1) & (df["y"] == 0)])
            numerator_unpriv_client_2 = len(df[(df[sensitive_attribute] == 0) & (df["y"] == 0) & (df["y_pred"] == 1)])
            numerator_priv_client_2 = len(df[(df[sensitive_attribute] == 1) & (df["y"] == 0) & (df["y_pred"] == 1)])
            denominator_unpriv_client_array = [denominator_unpriv_client, denominator_unpriv_client_2]
            denominator_priv_client_array = [denominator_priv_client, denominator_priv_client_2]
            numerator_unpriv_client_array = [numerator_unpriv_client, numerator_unpriv_client_2]
            numerator_priv_client_array = [numerator_priv_client, numerator_priv_client_2]
    elif metric == "SP":
        denominator_unpriv_client = len(df[(df[sensitive_attribute] == 0)])
        denominator_priv_client = len(df[(df[sensitive_attribute] == 1)])
        numerator_unpriv_client = len(df[(df[sensitive_attribute] == 0) & (df["y_pred"] == 1)])
        numerator_priv_client = len(df[(df[sensitive_attribute] == 1) & (df["y_pred"] == 1)])
        denominator_unpriv_client_array = [denominator_unpriv_client]
        denominator_priv_client_array = [denominator_priv_client]
        numerator_unpriv_client_array = [numerator_unpriv_client]
        numerator_priv_client_array = [numerator_priv_client]
    else:
        exit("Metric not supported")

    return denominator_unpriv_client_array, denominator_priv_client_array, numerator_unpriv_client_array, numerator_priv_client_array


def calculate_global_fairness(n, denominator_unpriv, denominator_priv, numerator_unpriv, numerator_priv):
    number_of_metrics = len(numerator_unpriv)
    number_of_clients = len(denominator_unpriv[0])
    global_fairness = 0

    for j in range(number_of_metrics):
        global_fairness_iter = []

        for i in range(number_of_clients):
            sum_denominator_unpriv = sum([denominator_unpriv[j][i] for j in range(len(denominator_unpriv))])
            sum_denominator_priv = sum([denominator_priv[j][i] for j in range(len(denominator_priv))])

            if sum_denominator_unpriv == 0 or sum_denominator_priv == 0:
                return -1

            if numerator_unpriv[j][i] != 0 and numerator_priv[j][i] != 0:
                unpriv_global_fairness = (numerator_unpriv[j][i] / sum_denominator_unpriv)
                priv_global_fairness = (numerator_priv[j][i] / sum_denominator_priv)
                global_fairness_iter.append((n[j] / sum(n)) * unpriv_global_fairness / priv_global_fairness)

        if len(global_fairness_iter) != 0:
            global_fairness += sum(global_fairness_iter) / len(global_fairness_iter)

    return global_fairness
