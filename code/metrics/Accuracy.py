from code.metrics.Metric import Metric

import numpy as np


class Accuracy(Metric):
    def __init__(self, name):
        self.name = name
        super().__init__(name)

    def calculate(self, _, df):
        y_pred = np.greater_equal(df["y_pred"], 0.5)
        y_true = np.greater_equal(df["y"], 0.5)
        value = round((y_pred == y_true).mean(), 2)
        self.values.append(value)

        print("{} - {}".format(self.name, value))
