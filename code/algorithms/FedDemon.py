from code.tensorflow.server.FedDemonAggregation import FedDemonAggregation
from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.tensorflow.client.FLClientSide import FLClientSide


class FedDemon(FederatedLearningAlgorithm):

    def __init__(self):
        super().__init__("fed_demon")
        self.ffm = None

    def get_filename(self, hyperparameters):
        return "b_{}".format(str(hyperparameters.beta0))

    def reset(self, federated_train_data, seed, hyperparameters, dataset):
        algorithm = FLClientSide(
            False, federated_train_data, dataset.n_features, dataset.learning_rate, seed
        )
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

        self.ffm = FedDemonAggregation(state, dataset, beta0=hyperparameters.beta0)

    def update(self, weights, x_val, y_val, clients_data_size, dataset):
        return self.ffm.update_model(weights, dataset.n_features, clients_data_size)
