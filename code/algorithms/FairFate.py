from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.FLClientSide import FederatedLearningClientSide
from code.tensorflow.FederatedFairMomentum import FederatedFairMomentum


class FairFate(FederatedLearningAlgorithm):
    def __init__(self, dataset, beta, lambda_init, aggregation_metrics):
        name = "fair_fate"
        hyperparameter_specs_str = get_hyperparameter_specs_str(beta, lambda_init, aggregation_metrics)
        super().__init__(name, hyperparameter_specs_str)

        self.dataset = dataset
        self.beta = beta
        self.lambda_init = lambda_init
        self.aggregation_metrics = aggregation_metrics
        self.ffm = None

    def reset(self, federated_train_data, seed):
        algorithm = FederatedLearningClientSide(False, federated_train_data, self.dataset.n_features, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in self.aggregation_metrics:
            metric.reset()
        self.ffm = FederatedFairMomentum(state, self.dataset, self.aggregation_metrics, beta=self.beta, lambda_init=self.lambda_init)

    def update(self, weights, x_val, y_val, clients_data_size):
        print("\nLambda: {}".format(round(self.ffm.lambda_, 2)))

        return self.ffm.update_model(weights, self.dataset.n_features, x_val, y_val, clients_data_size)


def get_hyperparameter_specs_str(beta, lambda_init, aggregation_metrics):
    aggregation_metrics_string = "-".join([metric.name for metric in aggregation_metrics])
    lambda_ = "e{}".format(lambda_init)

    return "l_{}_b_{}_{}".format(str(lambda_), str(beta), aggregation_metrics_string)
