from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.algorithms.FedAvg import fed_avg_update
from code.tensorflow.FLClientSide import FederatedLearningClientSide


class FedAvgLR(FederatedLearningAlgorithm):  # Local Reweighting
    def __init__(self, federated_train_data, n_features, seed):
        name = "fedavg_lr"
        algorithm = FederatedLearningClientSide("LR", federated_train_data, n_features, seed)
        state = algorithm.initialize()
        super().__init__(name, algorithm, state)

    def update(self, weights, n_features, unused_x_val, unused_y_val, clients_data_size):
        return fed_avg_update(weights, n_features, clients_data_size)
