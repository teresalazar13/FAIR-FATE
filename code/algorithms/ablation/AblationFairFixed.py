from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.ablation.AblationFairFixedAggregation import AblationFairFixedAggregation


# same as FAIR-FATE but with no momentum instead of decay and fixed lambda
# does not have beta0
# does not have rho, max and l0 but has l
class AblationFairFixed(FederatedLearningAlgorithm):

    def __init__(self):
        self.ffm = None
        super().__init__("ablation_fair_fixed")

    def get_hyperparameter_str(self, hyperparameters):
        aggregation_metrics_string = "-".join([metric.name for metric in hyperparameters.aggregation_metrics])

        return "l-{}_{}".format(str(hyperparameters.l), aggregation_metrics_string)

    def reset(self, federated_train_data, seed, hyperparameters, dataset):
        algorithm = FLClientSide(False, federated_train_data, dataset.n_features, dataset.learning_rate, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in hyperparameters.aggregation_metrics:
            metric.reset()
        self.ffm = AblationFairFixedAggregation(
            state, dataset, hyperparameters.aggregation_metrics, l=hyperparameters.l
        )

    def update(self, weights, x_val, y_val, clients_data_size, dataset):
        return self.ffm.update_model(weights, dataset.n_features, x_val, y_val, clients_data_size)
