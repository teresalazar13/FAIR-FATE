from code.algorithms.FLAlgorithm import FederatedLearningAlgorithm
from code.algorithms.FedAvg import fed_avg_update
from code.tensorflow.FLClientSide import FederatedLearningClientSide


class FedAvgGR(FederatedLearningAlgorithm):  # Global Reweighting
    def __init__(self, federated_train_data, x_train):
        name = "fedavg_gr"
        algorithm = FederatedLearningClientSide(2, federated_train_data, x_train[0])
        state = algorithm.initialize()
        super().__init__(name, algorithm, state)

    def update(self, weights, x_train, unused_x_val, unused_y_val):
        return fed_avg_update(weights, x_train)
