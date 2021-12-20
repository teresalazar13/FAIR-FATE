from code.metrics.Metric import Metric


class F1Score(Metric):
    def __init__(self, name):
        self.name = name
        super().__init__(name)

    def calculate(self, _, df):
        TP = len(df[(df["y"] == 1) & (df["y_pred"] == 1)])
        FP = len(df[(df["y"] == 0) & (df["y_pred"] == 1)])
        FN = len(df[(df["y"] == 1) & (df["y_pred"] == 0)])
        value = round((2 * TP) / ((2 * TP) + FP + FN), 2)
        self.values.append(value)

        print("{} - {}".format(self.name, value))
