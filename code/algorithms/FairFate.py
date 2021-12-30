from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.FLClientSide import FederatedLearningClientSide
from code.tensorflow.FederatedFairMomentum import FederatedFairMomentum


class FairFate(FederatedLearningAlgorithm):
    def __init__(self, federated_train_data, n_features, seed, dataset, aggregation_metrics, lambda_exponential=None, lambda_fixed=None, beta=.9):
        name = "fair_fate"
        algorithm = FederatedLearningClientSide(False, federated_train_data, n_features, seed)
        state = algorithm.initialize()
        aggregation_metrics_string = "-".join([metric.name for metric in aggregation_metrics])
        lambda_ = "e{}".format(lambda_exponential)
        if not lambda_exponential:
            lambda_ = "f{}".format(lambda_fixed)
        hyperparameter_specs_str = "l_{}_b_{}_{}".format(str(lambda_), str(beta), aggregation_metrics_string)
        super().__init__(name, algorithm, state, hyperparameter_specs_str)

        self.ffm = FederatedFairMomentum(state, dataset, aggregation_metrics, beta=beta, lambda_exponential=lambda_exponential, lambda_fixed=lambda_fixed)

    def update(self, weights, n_features, x_val, y_val, clients_data_size):
        print("\nLambda: {}".format(round(self.ffm.lambda_, 2)))

        return self.ffm.update_model(weights, n_features, x_val, y_val, clients_data_size)
