from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.ablation.AblationFairDemonLinearAggregation import AblationFairDemonLinearAggregation


# same as FAIR-FATE but with linear instead of exponential growth
# does not have rho, but has eta hyperparameter
class AblationFairDemonLinear(FederatedLearningAlgorithm):

    def __init__(self):
        super().__init__("ablation_fair_demon_linear")
        self.ffm = None

    def get_filename(self, hyperparameters):
        aggregation_metrics_string = "-".join([metric.name for metric in hyperparameters.aggregation_metrics])

        return "b0-{}_eta-{}_l0-{}_max-{}_{}".format(
            str(hyperparameters.beta0), str(hyperparameters.eta), str(hyperparameters.l0), str(hyperparameters.MAX),
            aggregation_metrics_string
        )

    def reset(self, federated_train_data, seed, hyperparameters, dataset):
        algorithm = FLClientSide(False, federated_train_data, dataset.n_features, dataset.learning_rate, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in hyperparameters.aggregation_metrics:
            metric.reset()
        self.ffm = AblationFairDemonLinearAggregation(
            state, dataset, hyperparameters.aggregation_metrics, beta0=hyperparameters.beta0, eta=hyperparameters.eta,
            MAX=hyperparameters.MAX, l0=hyperparameters.l0
        )

    def update(self, weights, x_val, y_val, clients_data_size, _, dataset):
        print("\nLambda: {}".format(round(self.ffm.lambda_, 2)))
        print("Beta: {}".format(round(self.ffm.beta, 2)))

        return self.ffm.update_model(weights, dataset.n_features, x_val, y_val, clients_data_size)
