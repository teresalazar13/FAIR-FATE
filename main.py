from code.algorithms.AlgorithmFactory import get_algorithm, get_algorithms_names
from code.algorithms.Hyperparameters import Hyperparameters
from code.datasets.DatasetFactory import get_dataset, get_datasets_names
from code.plots.pie_chart import create_stats_sensitive_distribution_all
from code.run import run

import sys
import argparse


def main(args):
    dataset = get_dataset(args.dataset)
    create_stats_sensitive_distribution_all(dataset, "./datasets/{}".format(dataset.name))
    alpha = float(args.alpha) if args.alpha else None
    fl = get_algorithm(args.fl)
    n_runs = int(args.n_runs)
    n_rounds = int(args.n_rounds)
    run(dataset, Hyperparameters(args), n_rounds, n_runs, fl, alpha=alpha)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=get_datasets_names(), required=True, help='dataset name')
    parser.add_argument('--fl', choices=get_algorithms_names(), required=True, help='Federated Learning algorithm')
    parser.add_argument('--n_runs', required=True, help='n_runs')
    parser.add_argument('--n_rounds', required=True, help='n_rounds')
    parser.add_argument('--alpha', required=False, help='alpha')
    parser.add_argument('--beta', required=False, help='beta')
    parser.add_argument('--beta0', required=False, help='beta0')
    parser.add_argument('--rho', required=False, help='rho')
    parser.add_argument('--eta', required=False, help='eta')
    parser.add_argument('--l0', required=False, help='l0')
    parser.add_argument('--l', required=False, help='l')
    parser.add_argument('--MAX', required=False, help='MAX')
    parser.add_argument('--metrics', required=False, help='metrics', nargs='+')

    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    main(get_arguments())
