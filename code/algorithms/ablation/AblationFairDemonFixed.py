from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.ablation.AblationFairDemonFixedAggregation import AblationFairDemonFixedAggregation


# same as FAIR-FATE but with fixed lambda
# does not have rho, l0 or MAX, but has l
class AblationFairDemonLinear(FederatedLearningAlgorithm):

    def __init__(self, dataset, aggregation_metrics, beta0, l):
        name = "ablation_fair_demon_fixed"
        hyperparameter_specs_str = get_hyperparameter_specs_str(beta0, l, aggregation_metrics)
        super().__init__(name, hyperparameter_specs_str)

        self.dataset = dataset
        self.aggregation_metrics = aggregation_metrics
        self.beta0 = beta0
        self.l = l
        self.ffm = None

    def reset(self, federated_train_data, seed):
        algorithm = FLClientSide(False, federated_train_data, self.dataset.n_features, self.dataset.learning_rate, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in self.aggregation_metrics:
            metric.reset()
        self.ffm = AblationFairDemonFixedAggregation(
            state, self.dataset, self.aggregation_metrics, beta0=self.beta0, l=self.l
        )

    def update(self, weights, x_val, y_val, clients_data_size, _):
        print("\nBeta: {}".format(round(self.ffm.beta, 2)))

        return self.ffm.update_model(weights, self.dataset.n_features, x_val, y_val, clients_data_size)


def get_hyperparameter_specs_str(beta0, l, aggregation_metrics):
    aggregation_metrics_string = "-".join([metric.name for metric in aggregation_metrics])

    return "b0-{}_l-{}_{}".format(str(beta0), str(l), aggregation_metrics_string)
