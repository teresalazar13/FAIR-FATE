from code.tensorflow.FederatedMomentum import FederatedMomentum
from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.FLClientSide import FederatedLearningClientSide


class FedMom(FederatedLearningAlgorithm):
    def __init__(self, dataset, beta):
        name = "fedmom"
        hyperparameter_specs_str = "b_{}".format(str(beta))
        super().__init__(name, hyperparameter_specs_str)

        self.dataset = dataset
        self.beta = beta
        self.ffm = None

    def reset(self, federated_train_data, seed):
        algorithm = FederatedLearningClientSide(False, federated_train_data, self.dataset.n_features, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        self.ffm = FederatedMomentum(state, self.dataset, beta=self.beta)

    def update(self, weights, x_val, y_val, clients_data_size):
        return self.ffm.update_model(weights, self.dataset.n_features, clients_data_size)
