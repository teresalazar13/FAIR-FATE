from code.tensorflow.FederatedMomentum import FederatedMomentum
from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.FLClientSide import FederatedLearningClientSide


class FedMom(FederatedLearningAlgorithm):
    def __init__(self, federated_train_data, x_train, dataset, beta=.5):
        name = "fedmom"
        algorithm = FederatedLearningClientSide(False, federated_train_data, x_train[0])
        state = algorithm.initialize()
        hyperparameter_specs_str = "b_{}".format(str(beta))
        super().__init__(name, algorithm, state, hyperparameter_specs_str)

        self.ffm = FederatedMomentum(state, dataset, beta=beta)

    def update(self, weights, x_train, x_val, y_val):
        return self.ffm.update_model(weights, x_train, x_val, y_val)
