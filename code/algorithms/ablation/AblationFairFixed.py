from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.ablation.AblationFairFixedAggregation import AblationFairFixedAggregation


# same as FAIR-FATE but with no momentum instead of decay and fixed lambda
# does not have beta0
# does not have rho, max and l0 but has l
class AblationFairLinear(FederatedLearningAlgorithm):

    def __init__(self, dataset, l, aggregation_metrics):
        name = "ablation_fair_fixed"
        hyperparameter_specs_str = get_hyperparameter_specs_str(l, aggregation_metrics)
        super().__init__(name, hyperparameter_specs_str)

        self.dataset = dataset
        self.aggregation_metrics = aggregation_metrics
        self.l = l
        self.ffm = None

    def reset(self, federated_train_data, seed):
        algorithm = FLClientSide(False, federated_train_data, self.dataset.n_features, self.dataset.learning_rate, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in self.aggregation_metrics:
            metric.reset()
        self.ffm = AblationFairFixedAggregation(state, self.dataset, self.aggregation_metrics, l=self.l)

    def update(self, weights, x_val, y_val, clients_data_size, _):
        return self.ffm.update_model(weights, self.dataset.n_features, x_val, y_val, clients_data_size)


def get_hyperparameter_specs_str(l, aggregation_metrics):
    aggregation_metrics_string = "-".join([metric.name for metric in aggregation_metrics])

    return "l-{}_{}".format(str(l), aggregation_metrics_string)
