from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.ablation.AblationFairMomFixedAggregation import AblationFairMomFixedAggregation


# same as FAIR-FATE but with fixed lambda and fixed beta
# does not have rho, l0 or MAX, but has l
# does not have beta0 but has beta
class AblationFairMomLinear(FederatedLearningAlgorithm):

    def __init__(self, dataset, aggregation_metrics, beta, l):
        name = "ablation_fair_mom_fixed"
        hyperparameter_specs_str = get_hyperparameter_specs_str(beta, l, aggregation_metrics)
        super().__init__(name, hyperparameter_specs_str)

        self.dataset = dataset
        self.aggregation_metrics = aggregation_metrics
        self.beta = beta
        self.l = l
        self.ffm = None

    def reset(self, federated_train_data, seed):
        algorithm = FLClientSide(False, federated_train_data, self.dataset.n_features, self.dataset.learning_rate, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in self.aggregation_metrics:
            metric.reset()
        self.ffm = AblationFairMomFixedAggregation(
            state, self.dataset, self.aggregation_metrics, beta=self.beta, l=self.l
        )

    def update(self, weights, x_val, y_val, clients_data_size, _):
        return self.ffm.update_model(weights, self.dataset.n_features, x_val, y_val, clients_data_size)


def get_hyperparameter_specs_str(beta0, l, aggregation_metrics):
    aggregation_metrics_string = "-".join([metric.name for metric in aggregation_metrics])

    return "b-{}_l-{}_{}".format(str(beta0), str(l), aggregation_metrics_string)
