from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.FLClientSide import FederatedLearningClientSide
from code.tensorflow.FederatedFairMomentum import FederatedFairMomentum


class FairFate(FederatedLearningAlgorithm):
    def __init__(self, federated_train_data, x_train, dataset, aggregation_metrics, lambda_=.5, beta=.9):
        name = "fair_fate"
        algorithm = FederatedLearningClientSide(False, federated_train_data, x_train[0])
        state = algorithm.initialize()
        hyperparameter_specs_str = "l_{}_b_{}".format(str(lambda_), str(beta))
        super().__init__(name, algorithm, state, hyperparameter_specs_str)

        self.ffm = FederatedFairMomentum(state, dataset, aggregation_metrics, beta=beta, lambda_=lambda_)

    def update(self, weights, x_train, x_val, y_val):
        return self.ffm.update_model(weights, x_train, x_val, y_val)
