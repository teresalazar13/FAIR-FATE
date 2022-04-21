from code.datasets.Adult import Adult
from code.datasets.Compas import Compas
from code.datasets.Law import Law
from code.datasets.Dutch import Dutch
from code.plots.pie_chart import create_stats_sensitive_distribution_all
from code.metrics.GroupBasedMetric import GroupBasedMetric, PosSens, Sens, TP, FN, FP, TN
from code.metrics.SuperGroupBasedMetric import SuperGroupBasedMetric
from code.run import run
from code.algorithms.FairFate import FairFate
from code.algorithms.FedAvg import FedAvg
from code.algorithms.FedAvgLR import FedAvgLR
from code.algorithms.FedAvgGR import FedAvgGR
from code.algorithms.FedMom import FedMom
from code.algorithms.FedVal import FedVal
from code.algorithms.FairFed import FairFed

import sys
import argparse


def main(args):
    dataset = get_dataset(args.dataset)
    create_stats_sensitive_distribution_all(dataset, "./datasets/{}".format(dataset.name))
    alpha = get_alpha(args.alpha)
    fl = get_fl(dataset, args.fl, args.beta, args.rho, args.l0, args.MAX, args.metrics)
    n_runs = int(args.n_runs)
    n_rounds = int(args.n_rounds)
    run(dataset, n_rounds, n_runs, fl, alpha=alpha)


def get_dataset(dataset_name):
    dataset = None

    if dataset_name == "adult":
        dataset = Adult()
    elif dataset_name == "compas":
        dataset = Compas()
    elif dataset_name == "law":
        dataset = Law()
    elif dataset_name == "dutch":
        dataset = Dutch()

    return dataset


def get_alpha(alpha_string):
    if alpha_string:
        alpha = float(alpha_string)
    else:
        alpha = None

    return alpha


def get_fl(dataset, fl_string, beta_string, rho_string, l0_string, MAX_string, metrics_string_array):
    fl = None

    if fl_string == "FedAvg":
        fl = FedAvg(dataset)
    elif fl_string == "FedAvgLR":
        fl = FedAvgLR(dataset)
    elif fl_string == "FedAvgGR":
        fl = FedAvgGR(dataset)
    elif fl_string == "FedVal":
        aggregation_metrics = get_aggregation_metrics(metrics_string_array)
        fl = FedVal(dataset, aggregation_metrics)
    elif fl_string == "FairFed":
        aggregation_metrics = get_aggregation_metrics(metrics_string_array)
        beta = float(beta_string)
        fl = FairFed(dataset, aggregation_metrics, beta)
    else:
        beta = float(beta_string)
        if fl_string == "FedMom":
            fl = FedMom(dataset, beta)
        elif fl_string == "FairFate":
            rho = float(rho_string)
            if l0_string:
                l0 = float(l0_string)
            else:
                l0 = 0.1  # default
            if MAX_string:
                MAX = float(MAX_string)
            else:
                MAX = 10000  # default
            aggregation_metrics = get_aggregation_metrics(metrics_string_array)
            fl = FairFate(dataset, beta, rho, l0, MAX, aggregation_metrics)

    return fl


def get_aggregation_metrics(metrics_string_array):
    aggregation_metrics = []
    for metric_string in metrics_string_array:
        if metric_string == "SP":
            aggregation_metrics.append(GroupBasedMetric("SP", PosSens(), Sens()))
        elif metric_string == "TPR":
            aggregation_metrics.append(GroupBasedMetric("TPR", TP(), FN()))
        elif metric_string == "FPR":
            aggregation_metrics.append(GroupBasedMetric("FPR", FP(), TN()))
        elif metric_string == "EQO":
            aggregation_metrics.append(SuperGroupBasedMetric("EQO", [GroupBasedMetric("TPR", TP(), FN()), GroupBasedMetric("FPR", FP(), TN())]))

    return aggregation_metrics


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["adult", "compas", "law", "dutch"], required=True, help='dataset name')
    parser.add_argument('--fl', choices=["FedAvg", "FedAvgLR", "FedAvgGR", "FedMom", "FedVal", "FairFed", "FairFate"], required=True,
                        help='Federated Learning algorithm')
    parser.add_argument('--alpha', required=False, help='alpha')
    parser.add_argument('--beta', required=False, help='beta')
    parser.add_argument('--rho', required=False, help='rho')
    parser.add_argument('--l0', required=False, help='l0')
    parser.add_argument('--MAX', required=False, help='MAX')
    parser.add_argument('--metrics', required=False, help='metrics', nargs='+')
    parser.add_argument('--n_runs', required=True, help='n_runs')
    parser.add_argument('--n_rounds', required=True, help='n_rounds')

    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    main(get_arguments())
