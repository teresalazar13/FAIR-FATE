from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.algorithms.FedAvg import fed_avg_update
from code.tensorflow.FLClientSide import FederatedLearningClientSide


class FedAvgLR(FederatedLearningAlgorithm):  # Local Reweighting
    def __init__(self, dataset):
        name = "fedavg_lr"
        super().__init__(name)

        self.dataset = dataset

    def reset(self, federated_train_data, seed):
        algorithm = FederatedLearningClientSide(
            "LR", federated_train_data, self.dataset.n_features, self.dataset.learning_rate, seed
        )
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

    def update(self, weights, unused_x_val, unused_y_val, clients_data_size, _):
        return fed_avg_update(weights, self.dataset.n_features, clients_data_size)
