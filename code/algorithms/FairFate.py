from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.FLClientSide import FederatedLearningClientSide
from code.tensorflow.FederatedFairMomentum import FederatedFairMomentum


class FairFate(FederatedLearningAlgorithm):
    def __init__(self, dataset, beta, rho, aggregation_metrics):
        name = "fair_fate"
        hyperparameter_specs_str = get_hyperparameter_specs_str(beta, rho, aggregation_metrics)
        super().__init__(name, hyperparameter_specs_str)

        self.dataset = dataset
        self.beta = beta
        self.rho = rho
        self.aggregation_metrics = aggregation_metrics
        self.ffm = None

    def reset(self, federated_train_data, seed):
        algorithm = FederatedLearningClientSide(
            False, federated_train_data, self.dataset.n_features, self.dataset.learning_rate, seed
        )
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in self.aggregation_metrics:
            metric.reset()
        self.ffm = FederatedFairMomentum(state, self.dataset, self.aggregation_metrics, beta=self.beta, rho=self.rho)

    def update(self, weights, x_val, y_val, clients_data_size, _):
        print("\nLambda: {}".format(round(self.ffm.lambda_, 2)))

        return self.ffm.update_model(weights, self.dataset.n_features, x_val, y_val, clients_data_size)


def get_hyperparameter_specs_str(beta, rho, aggregation_metrics):
    aggregation_metrics_string = "-".join([metric.name for metric in aggregation_metrics])
    rho_ = "e{}".format(rho)

    return "l_{}_b_{}_{}".format(str(rho_), str(beta), aggregation_metrics_string)
