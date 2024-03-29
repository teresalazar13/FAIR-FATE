from code.metrics.Accuracy import Accuracy
from code.metrics.F1Score import F1Score
from code.metrics.MCC import MCC
from code.metrics.SuperGroupBasedMetric import SuperGroupBasedMetric
from code.metrics.GroupBasedMetric import GroupBasedMetric, TN, TP, FN, FP, PosSens, Sens

import numpy as np
import pandas as pd


def create_metrics():
    ACC = Accuracy("ACC")
    f1Score = F1Score("F1Score")
    mcc = MCC("MCC")
    SP = GroupBasedMetric("SP", PosSens(), Sens())
    TPR = GroupBasedMetric("TPR", TP(), FN())
    TNR = GroupBasedMetric("TNR", TN(), FP())
    FPR = GroupBasedMetric("FPR", FP(), TN())
    FNR = GroupBasedMetric("FNR", FN(), TP())

    PPV = GroupBasedMetric("PPV", TP(), FP())
    NPV = GroupBasedMetric("NPV", TN(), FN())
    FDR = GroupBasedMetric("FDR", FP(), TP())
    FOR = GroupBasedMetric("FOR", FN(), TN())

    EQO = SuperGroupBasedMetric("EQO", [GroupBasedMetric("TPR", TP(), FN()), GroupBasedMetric("FPR", FP(), TN())])

    return [
        ACC, f1Score, mcc, SP,
        TPR, FPR, EQO,
        TNR, FNR, PPV, NPV, FDR, FOR
    ]


def get_aggregation_metrics(metrics_string_array):
    if not metrics_string_array:
        return None

    aggregation_metrics = []
    for metric_string in metrics_string_array:
        if metric_string == "SP":
            aggregation_metrics.append(GroupBasedMetric("SP", PosSens(), Sens()))
        elif metric_string == "TPR":
            aggregation_metrics.append(GroupBasedMetric("TPR", TP(), FN()))
        elif metric_string == "FPR":
            aggregation_metrics.append(GroupBasedMetric("FPR", FP(), TN()))
        elif metric_string == "EQO":
            aggregation_metrics.append(SuperGroupBasedMetric("EQO", [GroupBasedMetric("TPR", TP(), FN()),
                                                                     GroupBasedMetric("FPR", FP(), TN())]))

    return aggregation_metrics


def get_fairness(dataset, x_val, y_pred, y_val, metrics, debug=True):
    df = create_dataframe_for_eval(dataset.all_columns, x_val, y_pred, y_val)
    sensitive_attributes = [s.name for s in dataset.sensitive_attributes]

    metrics_values = []
    for metric in metrics:
        metrics_values.append(metric.calculate(sensitive_attributes, df, debug=debug)[0])

    return metrics_values


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
