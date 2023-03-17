from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.FedValAggregation import FedValAggregation
from code.metrics.Accuracy import Accuracy


class FedVal(FederatedLearningAlgorithm):

    def __init__(self, dataset, aggregation_metrics):
        hyperparameter_specs_str = "-".join([metric.name for metric in aggregation_metrics])
        super().__init__("fed_val", hyperparameter_specs_str)

        self.dataset = dataset
        #aggregation_metrics.append(Accuracy("ACC"))
        self.aggregation_metrics = aggregation_metrics
        self.ffm = None

    def reset(self, federated_train_data, seed):
        algorithm = FLClientSide(
            False, federated_train_data, self.dataset.n_features, self.dataset.learning_rate, seed
        )
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in self.aggregation_metrics:
            metric.reset()
        self.ffm = FedValAggregation(state, self.dataset, self.aggregation_metrics)

    def update(self, weights, x_val, y_val, clients_data_size, _):
        return self.ffm.update_model(weights, self.dataset.n_features, x_val, y_val)
