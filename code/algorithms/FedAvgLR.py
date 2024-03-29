from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.algorithms.FedAvg import fed_avg_update
from code.tensorflow.client.FLClientSide import FLClientSide


class FedAvgLR(FederatedLearningAlgorithm):  # Local Reweighting

    def __init__(self):
        super().__init__("fedavg_lr")

    def reset(self, federated_train_data, seed, _, dataset):
        algorithm = FLClientSide(
            "LR", federated_train_data, dataset.n_features, dataset.learning_rate, seed
        )
        state = algorithm.initialize()
        super().reset_algorithm(algorithm, state)

    def update(self, weights, unused_x_val, unused_y_val, clients_data_size, dataset):
        return fed_avg_update(weights, dataset.n_features, clients_data_size)
