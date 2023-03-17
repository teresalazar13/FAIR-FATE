from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.ablation.AblationFairDemonLinearAggregation import AblationFairDemonLinearAggregation


# same as FAIR-FATE but with linear instead of exponential growth
# does not have rho, but has eta hyperparameter
class AblationFairDemonLinear(FederatedLearningAlgorithm):

    def __init__(self, dataset, aggregation_metrics, beta0, eta, l0, MAX):
        name = "ablation_fair_demon_linear"
        hyperparameter_specs_str = get_hyperparameter_specs_str(beta0, eta, l0, MAX, aggregation_metrics)
        super().__init__(name, hyperparameter_specs_str)

        self.dataset = dataset
        self.aggregation_metrics = aggregation_metrics
        self.eta = eta
        self.beta0 = beta0
        self.l0 = l0
        self.MAX = MAX
        self.ffm = None

    def reset(self, federated_train_data, seed):
        algorithm = FLClientSide(False, federated_train_data, self.dataset.n_features, self.dataset.learning_rate, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in self.aggregation_metrics:
            metric.reset()
        self.ffm = AblationFairDemonLinearAggregation(
            state, self.dataset, self.aggregation_metrics, beta0=self.beta0, eta=self.eta, MAX=self.MAX, l0=self.l0
        )

    def update(self, weights, x_val, y_val, clients_data_size, _):
        print("\nLambda: {}".format(round(self.ffm.lambda_, 2)))
        print("Beta: {}".format(round(self.ffm.beta, 2)))

        return self.ffm.update_model(weights, self.dataset.n_features, x_val, y_val, clients_data_size)


def get_hyperparameter_specs_str(beta0, eta, l0, MAX, aggregation_metrics):
    aggregation_metrics_string = "-".join([metric.name for metric in aggregation_metrics])

    return "b0-{}_eta-{}_l0-{}_max-{}_{}".format(str(beta0), str(eta), str(l0), str(MAX), aggregation_metrics_string)
