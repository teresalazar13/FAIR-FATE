from code.tensorflow.server.FedMomAggregation import FedMomAggregation
from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide


class FedMom(FederatedLearningAlgorithm):

    def __init__(self, dataset, beta):
        hyperparameter_specs_str = "b_{}".format(str(beta))
        super().__init__("fedmom", hyperparameter_specs_str)

        self.dataset = dataset
        self.beta = beta
        self.ffm = None

    def reset(self, federated_train_data, seed):
        algorithm = FLClientSide(
            False, federated_train_data, self.dataset.n_features, self.dataset.learning_rate, seed
        )
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        self.ffm = FedMomAggregation(state, self.dataset, beta=self.beta)

    def update(self, weights, x_val, y_val, clients_data_size, _):
        return self.ffm.update_model(weights, self.dataset.n_features, clients_data_size)
