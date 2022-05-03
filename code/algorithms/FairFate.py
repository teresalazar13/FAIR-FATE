from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.FLClientSide import FederatedLearningClientSide
from code.tensorflow.FederatedFairMomentum import FederatedFairMomentum


class FairFate(FederatedLearningAlgorithm):
    def __init__(self, dataset, beta, rho, l0, MAX,  aggregation_metrics):
        name = "fair_fate"
        hyperparameter_specs_str = get_hyperparameter_specs_str(beta, rho, l0, MAX, aggregation_metrics)
        super().__init__(name, hyperparameter_specs_str)

        self.dataset = dataset
        self.beta = beta
        self.rho = rho
        self.l0 = l0
        self.MAX = MAX
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
        self.ffm = FederatedFairMomentum(state, self.dataset, self.aggregation_metrics, beta=self.beta, rho=self.rho, MAX=self.MAX, l0=self.l0)

    def update(self, weights, x_val, y_val, clients_data_size, _):
        print("\nLambda: {}".format(round(self.ffm.lambda_, 2)))
        print("Beta: {}".format(round(self.ffm.beta, 2)))

        return self.ffm.update_model(weights, self.dataset.n_features, x_val, y_val, clients_data_size)


def get_hyperparameter_specs_str(beta, rho, l0, MAX, aggregation_metrics):
    aggregation_metrics_string = "-".join([metric.name for metric in aggregation_metrics])

    return "b0-{}_rho-{}_l0-{}_max-{}_{}".format(str(beta), str(rho), str(l0), str(MAX), aggregation_metrics_string)
