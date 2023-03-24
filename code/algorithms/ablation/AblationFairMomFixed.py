from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide
from code.tensorflow.server.ablation.AblationFairMomFixedAggregation import AblationFairMomFixedAggregation


# same as FAIR-FATE but with fixed lambda and fixed beta
# does not have rho, l0 or MAX, but has l
# does not have beta0 but has beta
class AblationFairMomFixed(FederatedLearningAlgorithm):

    def __init__(self):
        super().__init__("ablation_fair_mom_fixed")
        self.ffm = None

    def get_filename(self, hyperparameters):
        aggregation_metrics_string = "-".join([metric.name for metric in hyperparameters.aggregation_metrics])

        return "b-{}_l-{}_{}".format(str(hyperparameters.beta0), str(hyperparameters.l), aggregation_metrics_string)

    def reset(self, federated_train_data, seed, hyperparameters, dataset):
        algorithm = FLClientSide(False, federated_train_data, dataset.n_features, dataset.learning_rate, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        for metric in hyperparameters.aggregation_metrics:
            metric.reset()
        self.ffm = AblationFairMomFixedAggregation(
            state, dataset, hyperparameters.aggregation_metrics, beta=hyperparameters.beta, l=hyperparameters.l
        )

    def update(self, weights, x_val, y_val, clients_data_size, _, dataset):
        return self.ffm.update_model(weights, dataset.n_features, x_val, y_val, clients_data_size)