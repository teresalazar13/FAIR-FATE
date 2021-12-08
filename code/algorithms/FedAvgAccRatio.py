from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.metrics.retriever import get_fairness, get_accuracy
from code.tensorflow.FLClientSide import FederatedLearningClientSide
from code.tensorflow.models import get_model

import numpy as np


class FedAvgAccRatio(FederatedLearningAlgorithm):
    def __init__(self, federated_train_data, x_train, dataset):
        name = "fedavg_fair_acc_ratio"
        algorithm = FederatedLearningClientSide(False, federated_train_data, x_train[0])
        state = algorithm.initialize()
        super().__init__(name, algorithm, state)

        self.dataset = dataset

    def update(self, weights, x_train, x_val, y_val):
        model = get_model(x_train)
        fair_acc_ratio = []
        for client_weights in weights:
            model.set_weights(client_weights)
            y_pred = model.predict(x_val)

            fairness = get_fairness(self.dataset, x_val, y_pred, y_val)
            if fairness == 0:
                fairness = 0.001
            acc = get_accuracy(y_pred, y_val)
            fair_acc_ratio.append(acc / fairness)

        new_state = []
        n_layers = len(weights[0])

        for layer in range(n_layers):
            weighted_sum = np.array(np.multiply(weights[0][layer], fair_acc_ratio[0] / sum(fair_acc_ratio)))
            for c in range(1, len(weights)):
                weighted_sum = np.add(
                    weighted_sum,
                    np.multiply(weights[c][layer], fair_acc_ratio[c] / sum(fair_acc_ratio))
                )
            new_state.append(weighted_sum)

        model.set_weights(new_state)

        return new_state, model
