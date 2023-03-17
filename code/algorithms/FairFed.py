from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.FairFedAggregation import FairFedAggregation


class FairFed(FederatedLearningAlgorithm):

    def __init__(self, dataset, aggregation_metrics, beta):
        hyperparameter_specs_str = "-".join([metric.name for metric in aggregation_metrics])
        super().__init__("fair_fed", hyperparameter_specs_str)

        self.beta = beta
        self.dataset = dataset
        self.aggregation_metrics = aggregation_metrics
        self.ffm = None

    def reset(self, federated_train_data, seed):
        algorithm = FLClientSide(
            "LR", federated_train_data, self.dataset.n_features, self.dataset.learning_rate, seed
        )
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in self.aggregation_metrics:
            metric.reset()
        self.ffm = FairFedAggregation(state, self.dataset, self.aggregation_metrics, self.beta)

    def update(self, weights, clients_dataset_x_y_label, _, clients_data_size, clients_idx):
        return self.ffm.update_model(weights, self.dataset.n_features, clients_dataset_x_y_label, clients_idx)
