from code.datasets.Adult import Adult
from code.datasets.Compas import Compas
from code.plot import plot
from code.plots.pie_chart import create_stats_sensitive_distribution_all
from code.metrics.GroupBasedMetric import GroupBasedMetric, PosSens, Sens, TP, FN, FP, TN
from code.metrics.SuperGroupBasedMetric import SuperGroupBasedMetric
from code.run import run
from code.algorithms.FairFate import FairFate
from code.algorithms.FedAvg import FedAvg
from code.algorithms.FedAvgLR import FedAvgLR
from code.algorithms.FedAvgGR import FedAvgGR
from code.algorithms.FedMom import FedMom

import sys
import argparse


def main(args):
    dataset = get_dataset(args.dataset)
    create_stats_sensitive_distribution_all(dataset, "./datasets/{}".format(dataset.name))
    alpha = get_alpha(args.alpha)
    fl = get_fl(dataset, args.fl, args.beta, args.lambda_, args.metrics)
    n_runs = int(args.n_runs)
    n_rounds = int(args.n_rounds)
    run(dataset, n_rounds, n_runs, fl, alpha=alpha)
    #plot(dataset.name, 10, ["SP_ratio"])


def get_dataset(dataset_name):
    dataset = None

    if dataset_name == "adult":
        dataset = Adult()
    elif dataset_name == "compas":
        dataset = Compas()

    return dataset


def get_alpha(alpha_string):
    if alpha_string:
        alpha = float(alpha_string)
    else:
        alpha = None

    return alpha


def get_fl(dataset, fl_string, beta_string, lambda_string, metrics_string_array):
    fl = None

    if fl_string == "FedAvg":
        fl = FedAvg(dataset)
    elif fl_string == "FedAvgLR":
        fl = FedAvgLR(dataset)
    elif fl_string == "FedAvgGR":
        fl = FedAvgGR(dataset)
    else:
        beta = float(beta_string)
        if fl_string == "FedMom":
            fl = FedMom(dataset, beta)
        elif fl_string == "FairFate":
            lambda_init = float(lambda_string)
            aggregation_metrics = get_aggregation_metrics(metrics_string_array)
            fl = FairFate(dataset, beta, lambda_init, aggregation_metrics)

    return fl


def get_aggregation_metrics(metrics_string_array):
    aggregation_metrics = []

    for metric_string in metrics_string_array:
        if metric_string == "SP":
            aggregation_metrics.append(GroupBasedMetric("SP", PosSens(), Sens()))
        elif metric_string == "TPR":
            aggregation_metrics.append(GroupBasedMetric("TPR", TP(), FN()))
        else:
            aggregation_metrics.append(SuperGroupBasedMetric("EQO", [GroupBasedMetric("TPR", TP(), FN()), GroupBasedMetric("FPR", FP(), TN())]))

    return aggregation_metrics



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["adult", "compas"], required=True, help='dataset name')
    parser.add_argument('--fl', choices=["FedAvg", "FedAvgLR", "FedAvgGR", "FedMom", "FairFate"], required=True,
                        help='Federated Learning algorithm')
    parser.add_argument('--alpha', required=False, help='alpha')
    parser.add_argument('--beta', required=False, help='beta')
    parser.add_argument('--lambda_', required=False, help='lambda')
    parser.add_argument('--metrics', required=False, help='metrics', nargs='+')
    parser.add_argument('--n_runs', required=True, help='n_runs')
    parser.add_argument('--n_rounds', required=True, help='n_rounds')

    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    main(get_arguments())
