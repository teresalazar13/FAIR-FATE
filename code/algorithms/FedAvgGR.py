from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.algorithms.FedAvg import fed_avg_update
from code.tensorflow.FLClientSide import FederatedLearningClientSide


class FedAvgGR(FederatedLearningAlgorithm):  # Global Reweighting
    def __init__(self, dataset):
        name = "fedavg_gr"
        super().__init__(name)

        self.dataset = dataset

    def reset(self, federated_train_data, seed):
        algorithm = FederatedLearningClientSide("GR", federated_train_data, self.dataset.n_features, seed)
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

    def update(self, weights, unused_x_val, unused_y_val, clients_data_size):
        return fed_avg_update(weights, self.dataset.n_features, clients_data_size)
