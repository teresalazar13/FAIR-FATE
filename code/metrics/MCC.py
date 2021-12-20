from code.metrics.Metric import Metric
import math


class MCC(Metric):
    def __init__(self, name):
        self.name = name
        super().__init__(name)

    def calculate(self, _, df):
        TP = len(df[(df["y"] == 1) & (df["y_pred"] == 1)])
        TN = len(df[(df["y"] == 0) & (df["y_pred"] == 0)])
        FP = len(df[(df["y"] == 0) & (df["y_pred"] == 1)])
        FN = len(df[(df["y"] == 1) & (df["y_pred"] == 0)])
        if (math.sqrt((TP+FP) * (TP + FN) * (TN + FP) * (TN + FN))) == 0:
            value = 0
        else:
            value = round((TP*TN - FP*FN) / (math.sqrt((TP+FP) * (TP + FN) * (TN + FP) * (TN + FN))), 2)
        self.values.append(value)

        print("{} - {}".format(self.name, value))
