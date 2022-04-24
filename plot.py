from code.plots.pareto_front import plot_pareto_fronts
from code.plots.plot_results_epochs import plot_results_epochs, plot_results_epochs_specific
from code.plots.plot import print_avg_results
from code.datasets.Adult import Adult
from code.datasets.Compas import Compas
from code.datasets.Law import Law
from code.datasets.Dutch import Dutch


def print_results(dataset_name, num_runs, n_rounds, fairness_metrics, alpha=None):
    metrics_results = ["ACC", "F1Score", "MCC", "SP_ratio", "TPR_ratio", "EQO_ratio"]
    metrics_results.extend(fairness_metrics)
    fls = ["fedavg", "fedavg_gr", "fedavg_lr"]
    fairness_metrics_string = "-".join([f.split("_")[0] for f in fairness_metrics])
    fl_fedval = "fed_val_{}".format(fairness_metrics_string)
    fls.append(fl_fedval)
    fl_fairfed = "fair_fed_{}".format(fairness_metrics_string)
    fls.append(fl_fairfed)

    fls_fair_fate = []
    for beta in [0.7, 0.8, 0.9, 0.99]:
        for rho in [0.035, 0.04, 0.045, 0.047, 0.05]:
            fl = "fair_fate_l_e{}_b_{}_{}".format(str(rho), str(beta), fairness_metrics_string)
            fls_fair_fate.append(fl)

    fls_fedmom = []
    for beta in [0.7, 0.8, 0.9, 0.99]:
        fl = "fedmom_b_{}".format(str(beta))
        fls_fedmom.append(fl)

    if alpha:
        for i in range(len(fls)):
            fls[i] = "{}_alpha-{}".format(fls[i], alpha)
        for i in range(len(fls_fair_fate)):
            fls_fair_fate[i] = "{}_alpha-{}".format(fls_fair_fate[i], alpha)
        for i in range(len(fls_fedmom)):
            fls_fedmom[i] = "{}_alpha-{}".format(fls_fedmom[i], alpha)

    print_avg_results(dataset_name, num_runs, n_rounds, fls, fls_fair_fate, fls_fedmom, fairness_metrics, metrics_results)


def plot_paretos(dataset_name, num_runs, num_rounds, alphas, metrics_F, metric_a, filename):
    rhos = [0.035, 0.04, 0.045, 0.047, 0.05]
    betas = [0.7, 0.8, 0.9, 0.99]
    fls_fair_fate = []

    for alpha in alphas:
        for metric_F in metrics_F:
            fls_fair_fate_alpha_metric = []
            for beta in betas:
                for rho in rhos:
                    fl = "fair_fate_l_e{}_b_{}_{}".format(str(rho), str(beta), metric_F[0].replace("_ratio", ""))
                    if alpha:
                        fl = "{}_alpha-{}".format(fl, alpha)
                    fls_fair_fate_alpha_metric.append(fl)
            fls_fair_fate.append([alpha, metric_F, fls_fair_fate_alpha_metric])

    plot_pareto_fronts(dataset_name, num_runs, num_rounds, fls_fair_fate, metric_a, rhos, betas, filename)


if __name__ == '__main__':
    alpha = 0.5
    dataset_name = Law().name
    metrics = ["TPR_ratio"]
    print_results(dataset_name, 20, 50, metrics, alpha=alpha)

    """
    plot_results_epochs(
        50, Dutch().name, 20, ["SP_ratio", "TPR_ratio", "EQO_ratio"],
        [0.5, 1.0, None],
        [[0.04, 0.04, 0.045], [0.045, 0.047, 0.047], [0.05, 0.047, 0.05]],
        [[0.9, 0.99, 0.7], [0.7, 0.8, 0.8], [0.99, 0.99, 0.99]],
        [0.8, 0.9, 0.9]
    )
    plot_results_epochs(
        50, Law().name, 20, ["SP_ratio", "TPR_ratio", "EQO_ratio"],
        [0.25, 0.5, None],
        [[0.05, 0.04, 0.045], [0.05, 0.047, 0.047], [0.05, 0.04, 0.045]],
        [[0.7, 0.7, 0.8], [0.8, 0.9, 0.7], [0.9, 0.8, 0.8]],
        [0.8, 0.7, 0.99]
    )
    plot_results_epochs(
        50, Adult().name, 20, ["SP_ratio", "TPR_ratio", "EQO_ratio"],
        [0.5, 1.0, None],
        [[0.045, 0.045, 0.045], [0.045, 0.047, 0.045], [0.05, 0.047, 0.05]],
        [[0.7, 0.8, 0.7], [0.7, 0.7, 0.7], [0.99, 0.99, 0.99]],
        [0.9, 0.9, 0.9]
    )
    plot_results_epochs(
        50, Compas().name, 20, ["SP_ratio", "TPR_ratio", "EQO_ratio"],
        [0.25, 0.5, None],
        [[0.045, 0.047, 0.045], [0.05, 0.045, 0.045], [0.05, 0.05, 0.05]],
        [[0.9, 0.8, 0.9], [0.7, 0.7, 0.7], [0.99, 0.99, 0.99]],
        [0.8, 0.9, 0.99]
    )"""

    """
    plot_results_epochs_specific(
        Compas().name, 20,
        [[
            "fedavg_alpha-0.25",
            "fedmom_b_0.8_alpha-0.25",
            "fedavg_gr_alpha-0.25",
            "fedavg_lr_alpha-0.25",
            "fed_val_TPR_alpha-0.25",
            "fair_fate_rho-0.047_l0-0.1_max-1.0_b-0.8_TPR_alpha-0.25"
        ],
        [
            "fedavg_alpha-0.5",
            "fedmom_b_0.9_alpha-0.5",
            "fedavg_gr_alpha-0.5",
            "fedavg_lr_alpha-0.5",
            "fed_val_TPR_alpha-0.5",
            "fair_fate_rho-0.045_l0-0.1_max-1.0_b-0.7_TPR_alpha-0.5"
        ],
        [
            "fedavg",
            "fedmom_b_0.99",
            "fedavg_gr",
            "fedavg_lr",
            "fed_val_TPR",
            "fair_fate_rho-0.05_l0-0.1_max-1.0_b-0.8_TPR"
        ]],
        [
            r'$\alpha=0.25$', r'$\alpha=0.5$', "RND",
        ],
        [
            "FedAvg",
            "FedMom",
            "FedAvg+GR",
            "FedAvg+LR",
            "FedVal (F=EO)",
            'FAIR-FATE (F=EO)'
        ]
    )"""

    #plot_paretos(Adult().name, 20, 50, [None, 1.0, 0.5], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-adult")
    #plot_paretos(Compas().name, 20, 50, [None, 0.5, 0.25], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-compas")
    #plot_paretos(Dutch().name, 20, 50, [None, 1.0, 0.5], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-dutch")
    #plot_paretos(Law().name, 20, 50, [None, 0.5, 0.25], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-law")
