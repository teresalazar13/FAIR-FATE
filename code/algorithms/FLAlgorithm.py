from code.metrics.retriever import create_metrics, calculate_metrics, get_metrics_as_df
from abc import abstractmethod


class FederatedLearningAlgorithm:
    def __init__(self, name, hyperparameter_specs_str=""):
        self.name = name
        self.hyperparameter_specs_str = hyperparameter_specs_str
        self.algorithm = None
        self.state = None
        self.metrics = None

    def reset_algorithm(self, algorithm, state):
        self.algorithm = algorithm
        self.state = state
        self.metrics = create_metrics()

    def iterate(self, dataset, federated_train_data, x_val, y_val, x_test, y_test, clients_data_size):
        self.state, weights = self.algorithm.next(self.state, federated_train_data)
        self.state, model = self.update(weights, x_val, y_val, clients_data_size)
        y_pred = model.predict(x_test)
        print("\n\n{}-{}\n".format(self.name, self.hyperparameter_specs_str))
        calculate_metrics(dataset, self.metrics, x_test, y_pred, y_test)

    @abstractmethod
    def update(self, weights, x_val, y_val, clients_data_size):
        raise NotImplementedError("Must override update")

    def save_metrics_to_file(self, dataset_name, run_num, alpha):
        filename_part = './datasets/{}/run_{}'.format(dataset_name, run_num)
        filename_part_name = self.name
        if self.hyperparameter_specs_str != "":
            filename_part_name += "_" + self.hyperparameter_specs_str
        if alpha:
            filename_part_name += "_alpha-" + str(alpha)

        filename = '{}/{}.csv'.format(filename_part, filename_part_name)

        df_metrics = get_metrics_as_df(self.metrics)
        df_metrics.to_csv(filename, index=False)
