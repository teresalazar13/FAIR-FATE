from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.ablation.AblationFairMomExponentialAggregation import AblationFairMomExponentialAggregation


# same as FAIR-FATE but with standard momentum instead of decay
# does not have beta0 but has beta
class AblationFairMomExponential(FederatedLearningAlgorithm):

    def __init__(self):
        self.ffm = None
        super().__init__("ablation_fair_mom_exponential")

    def get_hyperparameter_str(self, hyperparameters):
        aggregation_metrics_string = "-".join([metric.name for metric in hyperparameters.aggregation_metrics])

        return "b-{}_rho-{}_l0-{}_max-{}_{}".format(
            str(hyperparameters.beta), str(hyperparameters.rho), str(hyperparameters.l0), str(hyperparameters.MAX),
            aggregation_metrics_string
        )

    def reset(self, federated_train_data, seed, hyperparameters, dataset):
        algorithm = FLClientSide(False, federated_train_data, dataset.n_features, dataset.learning_rate, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in hyperparameters.aggregation_metrics:
            metric.reset()
        self.ffm = AblationFairMomExponentialAggregation(
            state, dataset, hyperparameters.aggregation_metrics, MAX=hyperparameters.MAX, l0=hyperparameters.l0,
            rho=hyperparameters.rho, beta=hyperparameters.beta
        )

    def update(self, weights, x_val, y_val, clients_data_size, dataset):
        print("\nLambda: {}".format(round(self.ffm.lambda_, 2)))

        return self.ffm.update_model(weights, dataset.n_features, x_val, y_val, clients_data_size)
