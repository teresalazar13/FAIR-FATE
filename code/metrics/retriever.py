from code.metrics.Accuracy import Accuracy
from code.metrics.GroupBasedMetric import GroupBasedMetric, TN, TP, FN, FP, PosSens, Sens

import numpy as np
import pandas as pd


def create_metrics():
    ACC = Accuracy("ACC")
    SP = GroupBasedMetric("SP", PosSens(), Sens())
    TPR = GroupBasedMetric("TPR", TP(), FN())
    TNR = GroupBasedMetric("TNR", TN(), FP())
    FPR = GroupBasedMetric("FPR", FP(), TN())
    FNR = GroupBasedMetric("FNR", FN(), TP())

    PPV = GroupBasedMetric("PPV", TP(), FP())
    NPV = GroupBasedMetric("NPV", TN(), FN())
    FDR = GroupBasedMetric("FDR", FP(), TP())
    FOR = GroupBasedMetric("FOR", FN(), TN())

    return [
        ACC, SP, TPR, TNR, FPR, FNR, PPV, NPV, FDR, FOR
    ]


def get_fairness(dataset, x_val, y_pred, y_val):  # Only works for SP now
    metric = GroupBasedMetric("SP", PosSens(), Sens())
    df = create_dataframe_for_eval(dataset.all_columns, x_val, y_pred, y_val)
    sensitive_attributes = [s.name for s in dataset.sensitive_attributes]
    metric.calculate(sensitive_attributes, df)

    return metric.ratios[0]


def get_accuracy(y_pred, y):
    y_pred = np.greater_equal(y_pred, 0.5)
    y_true = np.greater_equal(y, 0.5)

    return round((y_pred == y_true).mean(), 2)


def get_metrics_as_df(metrics):
    df = pd.DataFrame()

    for metric in metrics:
        if metric.ratios:
            name_ratio = metric.name + "_ratio"
            name_diff = metric.name + "_diff"
            values_ratio = metric.ratios
            values_diff = metric.differences
            df[name_ratio] = values_ratio
            df[name_diff] = values_diff

        else:
            df[metric.name] = metric.values

    return df


# Create df contain X features, y_pred and y_true columns
def create_dataframe_for_eval(all_columns, x_test, y_pred, y_test):
    y_pred = np.greater_equal(y_pred, 0.5)

    df = pd.DataFrame(data=np.concatenate((x_test, np.stack(y_test, axis=0), np.stack(y_pred, axis=0)), axis=1),
                      columns=all_columns + ["y", "y_pred"])

    return df


# Group-Based metrics
def calculate_metrics(dataset, metrics, x_test, y_pred, y_test):
    df = create_dataframe_for_eval(dataset.all_columns, x_test, y_pred, y_test)
    sensitive_attributes = [s.name for s in dataset.sensitive_attributes]

    for metric in metrics:
        metric.calculate(sensitive_attributes, df)
