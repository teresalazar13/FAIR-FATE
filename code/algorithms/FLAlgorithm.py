from code.metrics.retriever import create_metrics, calculate_metrics, get_metrics_as_df
from abc import abstractmethod


class FederatedLearningAlgorithm:
    def __init__(self, name, algorithm, state, hyperparameter_specs_str=""):
        self.name = name
        self.algorithm = algorithm
        self.state = state
        self.metrics = create_metrics()
        self.hyperparameter_specs_str = hyperparameter_specs_str

    def iterate(self, dataset, federated_train_data, x_train, x_val, y_val, x_test, y_test):
        self.state, weights = self.algorithm.next(self.state, federated_train_data)
        self.state, model = self.update(weights, x_train, x_val, y_val)
        y_pred = model.predict(x_test)
        print("\n\n" + self.name + "\n")
        calculate_metrics(dataset, self.metrics, x_test, y_pred, y_test)

    @abstractmethod
    def update(self, weights, x_train, x_val, y_val):
        raise NotImplementedError("Must override update")

    def save_metrics_to_file(self, dataset_name, run_num):
        filename_part = '/content/gdrive/MyDrive/Colab Notebooks/{}/run_{}'.format(dataset_name, run_num)

        if self.hyperparameter_specs_str == "":
            filename = '{}/{}.csv'.format(filename_part, self.name)
        else:
            filename = '{}/{}_{}.csv'.format(filename_part, self.name, self.hyperparameter_specs_str)

        df_metrics = get_metrics_as_df(self.metrics)
        df_metrics.to_csv(filename, index=False)
