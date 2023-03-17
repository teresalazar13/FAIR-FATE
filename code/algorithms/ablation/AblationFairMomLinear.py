from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.ablation.AblationFairMomLinearAggregation import AblationFairMomLinearAggregation


# same as FAIR-FATE but with standard momentum instead of decay and linear growth instead of exponential growth
# does not have beta0 but has beta
# does not have rho but has eta
class AblationFairMomLinear(FederatedLearningAlgorithm):

    def __init__(self, dataset, beta, eta, l0, MAX, aggregation_metrics):
        name = "ablation_fair_mom_linear"
        hyperparameter_specs_str = get_hyperparameter_specs_str(beta, eta, l0, MAX, aggregation_metrics)
        super().__init__(name, hyperparameter_specs_str)

        self.dataset = dataset
        self.aggregation_metrics = aggregation_metrics
        self.beta = beta
        self.eta = eta
        self.l0 = l0
        self.MAX = MAX
        self.ffm = None

    def reset(self, federated_train_data, seed):
        algorithm = FLClientSide(False, federated_train_data, self.dataset.n_features, self.dataset.learning_rate, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in self.aggregation_metrics:
            metric.reset()
        self.ffm = AblationFairMomLinearAggregation(
            state, self.dataset, self.aggregation_metrics, MAX=self.MAX, l0=self.l0, eta=self.eta, beta=self.beta
        )

    def update(self, weights, x_val, y_val, clients_data_size, _):
        print("\nLambda: {}".format(round(self.ffm.lambda_, 2)))

        return self.ffm.update_model(weights, self.dataset.n_features, x_val, y_val, clients_data_size)


def get_hyperparameter_specs_str(beta, eta, l0, MAX, aggregation_metrics):
    aggregation_metrics_string = "-".join([metric.name for metric in aggregation_metrics])

    return "b-{}_eta-{}_l0-{}_max-{}_{}".format(str(beta), str(eta), str(l0), str(MAX), aggregation_metrics_string)
