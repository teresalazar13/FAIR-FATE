from code.tensorflow.FederatedMomentum import FederatedMomentum
from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.FLClientSide import FederatedLearningClientSide


class FedMom(FederatedLearningAlgorithm):
    def __init__(self, federated_train_data, n_features, seed, dataset, beta=.5):
        name = "fedmom"
        algorithm = FederatedLearningClientSide(False, federated_train_data, n_features, seed)
        state = algorithm.initialize()
        hyperparameter_specs_str = "b_{}".format(str(beta))
        super().__init__(name, algorithm, state, hyperparameter_specs_str)

        self.ffm = FederatedMomentum(state, dataset, beta=beta)

    def update(self, weights, n_features, x_val, y_val, clients_data_size):
        return self.ffm.update_model(weights, n_features, clients_data_size)
