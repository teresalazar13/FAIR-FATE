from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.ablation.AblationFairExponentialAggregation import AblationFairExponentialAggregation


# same as FAIR-FATE but with no momentum instead of decay
# does not have beta0
class AblationFairExponential(FederatedLearningAlgorithm):

    def __init__(self, dataset, rho, l0, MAX, aggregation_metrics):
        name = "ablation_fair_exponential"
        hyperparameter_specs_str = get_hyperparameter_specs_str(rho, l0, MAX, aggregation_metrics)
        super().__init__(name, hyperparameter_specs_str)

        self.dataset = dataset
        self.aggregation_metrics = aggregation_metrics
        self.rho = rho
        self.l0 = l0
        self.MAX = MAX
        self.ffm = None

    def reset(self, federated_train_data, seed):
        algorithm = FLClientSide(False, federated_train_data, self.dataset.n_features, self.dataset.learning_rate, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in self.aggregation_metrics:
            metric.reset()
        self.ffm = AblationFairExponentialAggregation(
            state, self.dataset, self.aggregation_metrics, MAX=self.MAX, l0=self.l0, rho=self.rho
        )

    def update(self, weights, x_val, y_val, clients_data_size, _):
        print("\nLambda: {}".format(round(self.ffm.lambda_, 2)))

        return self.ffm.update_model(weights, self.dataset.n_features, x_val, y_val, clients_data_size)


def get_hyperparameter_specs_str(rho, l0, MAX, aggregation_metrics):
    aggregation_metrics_string = "-".join([metric.name for metric in aggregation_metrics])

    return "rho-{}_l0-{}_max-{}_{}".format(str(rho), str(l0), str(MAX), aggregation_metrics_string)
