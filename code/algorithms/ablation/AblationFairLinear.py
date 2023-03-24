from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.ablation.AblationFairLinearAggregation import AblationFairLinearAggregation


# same as FAIR-FATE but with no momentum instead of decay and linear growth
# does not have beta0
# does not have rho but has eta
class AblationFairLinear(FederatedLearningAlgorithm):

    def __init__(self):
        super().__init__("ablation_fair_linear")
        self.ffm = None

    def get_filename(self, hyperparameters):
        aggregation_metrics_string = "-".join([metric.name for metric in hyperparameters.aggregation_metrics])

        return "eta-{}_l0-{}_max-{}_{}".format(
            str(hyperparameters.eta), str(hyperparameters.l0), str(hyperparameters.MAX), aggregation_metrics_string
        )

    def reset(self, federated_train_data, seed, hyperparameters, dataset):
        algorithm = FLClientSide(False, federated_train_data, dataset.n_features, dataset.learning_rate, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in hyperparameters.aggregation_metrics:
            metric.reset()
        self.ffm = AblationFairLinearAggregation(
            state, dataset, hyperparameters.aggregation_metrics, MAX=hyperparameters.MAX, l0=hyperparameters.l0,
            eta=hyperparameters.eta
        )

    def update(self, weights, x_val, y_val, clients_data_size, dataset):
        print("\nLambda: {}".format(round(self.ffm.lambda_, 2)))

        return self.ffm.update_model(weights, dataset.n_features, x_val, y_val, clients_data_size)
