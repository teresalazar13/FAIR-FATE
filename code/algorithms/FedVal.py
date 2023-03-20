from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.FedValAggregation import FedValAggregation
from code.metrics.Accuracy import Accuracy


class FedVal(FederatedLearningAlgorithm):

    def __init__(self):
        super().__init__("fed_val")
        self.ffm = None
    def get_filename(self, hyperparameters):
        return "-".join([metric.name for metric in hyperparameters.aggregation_metrics])

    def reset(self, federated_train_data, seed, hyperparameters, dataset):
        algorithm = FLClientSide(
            False, federated_train_data, dataset.n_features, dataset.learning_rate, seed
        )
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in hyperparameters.aggregation_metrics:
            metric.reset()
        self.ffm = FedValAggregation(state, dataset, hyperparameters.aggregation_metrics)

    def update(self, weights, x_val, y_val, clients_data_size, _, dataset):
        return self.ffm.update_model(weights, dataset.n_features, x_val, y_val)
