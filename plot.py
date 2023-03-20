from code.plots.pareto_front import plot_pareto_fronts
from code.plots.plot_results_epochs import plot_results_epochs, plot_results_epochs_specific
from code.plots.plot import print_avg_results


def print_results(dataset_name, num_runs, n_rounds, fairness_metrics, alpha=None):
    metrics_results = ["ACC", "F1Score", "MCC", "SP_ratio", "TPR_ratio", "EQO_ratio"]
    metrics_results.extend(fairness_metrics)
    fls = ["fedavg", "fedavg_gr", "fedavg_lr"]
    fairness_metrics_string = "-".join([f.split("_")[0] for f in fairness_metrics])
    fl_fedval = "fed_val_{}".format(fairness_metrics_string)
    fls.append(fl_fedval)

    fls_fair_fate = []
    for b0 in [0.8, 0.9, 0.99]:
        for rho in [0.04, 0.05]:
            for l0 in [0.1, 0.5]:
                for max_ in [0.8, 0.9, 1.0]:
                    fl = "fair_fate_b0-{}_rho-{}_l0-{}_max-{}_{}".format(str(b0), str(rho), str(l0), str(max_), fairness_metrics_string)
                    fls_fair_fate.append(fl)

    fls_fedmom = []
    for beta in [0.8, 0.9, 0.99]:
        fl = "fedmom_b_{}".format(str(beta))
        fls_fedmom.append(fl)

    fls_fed_demon = []
    for beta in [0.8, 0.9, 0.99]:
        fl = "fed_demon_b_{}".format(str(beta))
        fls_fed_demon.append(fl)

    if alpha:
        for i in range(len(fls)):
            fls[i] = "{}_alpha-{}".format(fls[i], alpha)
        for i in range(len(fls_fair_fate)):
            fls_fair_fate[i] = "{}_alpha-{}".format(fls_fair_fate[i], alpha)
        for i in range(len(fls_fedmom)):
            fls_fedmom[i] = "{}_alpha-{}".format(fls_fedmom[i], alpha)

    print_avg_results(dataset_name, num_runs, n_rounds, fls, fls_fair_fate, fls_fedmom, fls_fed_demon, fairness_metrics, metrics_results)


def plot_paretos(
        dataset_name, num_runs, num_rounds, alphas, metrics_F, metric_a, hyperparameters, hyperparameter_name, filename
):
    fls_fair_fate = []

    for alpha in alphas:
        for metric_F in metrics_F:
            fls_fair_fate_alpha_metric = []
            for l0 in [0.1, 0.5]:
                for rho in [0.04, 0.05]:
                    for max in [0.8, 0.9, 1.0]:
                        for b0 in [0.8, 0.9, 0.99]:
                            fl = "fair_fate_b0-{}_rho-{}_l0-{}_max-{}_{}".format(str(b0), str(rho), str(l0), str(max), metric_F[0].replace("_ratio", ""))
                            if alpha:
                                fl = "{}_alpha-{}".format(fl, alpha)
                            fls_fair_fate_alpha_metric.append(fl)
            fls_fair_fate.append([alpha, metric_F, fls_fair_fate_alpha_metric])

    plot_pareto_fronts(
        dataset_name, num_runs, num_rounds, fls_fair_fate, metric_a, hyperparameters, hyperparameter_name, filename
    )


if __name__ == '__main__':  # TODO
    alpha = 0.5
    dataset_name = "compas"
    metrics = ["SP_ratio"]
    print_results(dataset_name, 10, 100, metrics, alpha=alpha)

    """
    plot_results_epochs(
        100, Compas().name, 10, ["SP_ratio", "TPR_ratio", "EQO_ratio"],
        [0.5, 1.0, None],
        [[0.8, 0.99, 0.99], [0.8, 0.8, 0.99], [0.9, 0.9, 0.9]],
        [[0.04, 0.05, 0.04], [0.04, 0.05, 0.05], [0.05, 0.05, 0.04]],
        [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]],
        [[0.9, 0.9, 0.9], [0.9, 0.8, 0.9], [1.0, 1.0, 1.0]],
        [0.8, 0.99, 0.8],
        [0.8, 0.8, 0.8]
    )
    plot_results_epochs(
        100, Adult().name, 10, ["SP_ratio", "TPR_ratio", "EQO_ratio"],
        [0.5, 1.0, None],
        [[0.8, 0.99, 0.9], [0.99, 0.8, 0.99], [0.99, 0.99, 0.9]],
        [[0.05, 0.04, 0.05], [0.04, 0.05, 0.04], [0.04, 0.04, 0.05]],
        [[0.1, 0.5, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.5]],
        [[0.8, 0.8, 0.9], [0.9, 1.0, 0.9], [1.0, 1.0, 1.0]],
        [0.8, 0.99, 0.99],
        [0.8, 0.8, 0.8]
    )
    plot_results_epochs(
        100, Law().name, 10, ["SP_ratio", "TPR_ratio", "EQO_ratio"],
        [0.5, 1.0, None],
        [[0.99, 0.9, 0.99], [0.8, 0.9, 0.9], [0.99, 0.99, 0.99]],
        [[0.05, 0.04, 0.04], [0.05, 0.05, 0.04], [0.05, 0.05, 0.05]],
        [[0.5, 0.5, 0.1], [0.1, 0.5, 0.5], [0.5, 0.5, 0.1]],
        [[1.0, 0.9, 0.8], [1.0, 1.0, 1.0], [1.0, 0.9, 1.0]],
        [0.8, 0.8, 0.99],
        [0.99, 0.99, 0.99]
    )
    plot_results_epochs(
        100, Dutch().name, 10, ["SP_ratio", "TPR_ratio", "EQO_ratio"],
        [0.5, 1.0, None],
        [[0.99, 0.8, 0.9], [0.8, 0.8, 0.8], [0.99, 0.99, 0.99]],
        [[0.04, 0.04, 0.04], [0.04, 0.05, 0.04], [0.05, 0.05, 0.04]],
        [[0.5, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.5, 0.1]],
        [[0.8, 0.8, 0.9], [0.8, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [0.8, 0.8, 0.8],
        [0.99, 0.99, 0.99]
    )"""

    """
    plot_results_epochs_specific(
        dataset_name, 1, 100, metrics[0].split("_")[0],
        [["fair_fate_b0-0.9_rho-0.05_l0-0.1_max-0.9_SP_alpha-0.5", "fedavg_alpha-0.5"]],
        ["", ""], ["", ""]
    )"""
    """
    hyperparameters_list = [[0.8, 0.9, 1.0], [0.1, 0.5], [0.8, 0.9, 0.99], [0.04, 0.05]]
    hyperparameter_name_list = [["MAX", "max"], ["\lambda_0", "l0"], ["\\beta_0", "b0"], ["\\rho", "rho"]]
    for i in range(len(hyperparameters_list)):
        hyperparameters = hyperparameters_list[i]
        hyperparameter_name = hyperparameter_name_list[i]

        plot_paretos(
            Adult().name, 10, 100, [0.5, 1.0, None], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC",
            hyperparameters, hyperparameter_name, "pareto_{}-{}".format(hyperparameter_name[1], Adult().name)
        )
        plot_paretos(
            Compas().name, 10, 100, [0.5, 1.0, None], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC",
            hyperparameters, hyperparameter_name, "pareto_{}-{}".format(hyperparameter_name[1], Compas().name)
        )
        plot_paretos(
            Dutch().name, 10, 100, [0.5, 1.0, None], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC",
            hyperparameters, hyperparameter_name, "pareto_{}-{}".format(hyperparameter_name[1], Dutch().name)
        )
        plot_paretos(
            Law().name, 10, 100, [0.5, 1.0, None], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC",
            hyperparameters, hyperparameter_name, "pareto_{}-{}".format(hyperparameter_name[1], Law().name)
        )"""
