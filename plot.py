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
    #fl_fairfed = "fair_fed_{}".format(fairness_metrics_string)
    #fls.append(fl_fairfed)

    fls_fair_fate = []
    #for beta in [0.7, 0.8, 0.9, 0.99]:
        #for rho in [0.035, 0.04, 0.045, 0.047, 0.05]:
            #fl = "fair_fate_l_e{}_b_{}_{}".format(str(rho), str(beta), fairness_metrics_string)
            #fls_fair_fate.append(fl)

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

    #fls_feddemon = []
    #for beta in [0.8, 0.9, 0.99]:
        #fl = "fedmom_b_{}".format(str(beta))
        #fls_fedmom.append(fl)

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
    metrics = ["EQO_ratio"]
    print_results(dataset_name, 10, 100, metrics, alpha=alpha)

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
    metric = "EQO"
    alpha = "_alpha-0.5"
    name = Law().name
    plot_results_epochs_specific(
        name, 1, 100, metric,
        [[
            #"fedavg{}".format(alpha),
            #"fedavg_lr{}".format(alpha),
            "fedavg_gr{}".format(alpha),

            #"fedmom_b_0.8{}".format(alpha),
            #"fedmom_b_0.85{}".format(alpha),
            #"fedmom_b_0.9{}".format(alpha),
            #"fedmom_b_0.95{}".format(alpha),
            #"fedmom_b_0.99{}".format(alpha),

            #"fed_demon_b_0.8{}".format(alpha),
            #"fed_demon_b_0.85{}".format(alpha),
            #"fed_demon_b_0.9{}".format(alpha),
            #"fed_demon_b_0.95{}".format(alpha),
            #"fed_demon_b_0.99{}".format(alpha),

            #"fed_val_{}{}".format(metric, alpha),

            #"fair_fate_b0-0.8_rho-0.04_l0-0.1_max-1.0_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.8_rho-0.05_l0-0.1_max-1.0_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.85_rho-0.04_l0-0.1_max-1.0_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.85_rho-0.05_l0-0.1_max-1.0_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.9_rho-0.04_l0-0.1_max-1.0_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.9_rho-0.05_l0-0.1_max-1.0_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.95_rho-0.04_l0-0.1_max-1.0_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.95_rho-0.05_l0-0.1_max-1.0_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.99_rho-0.04_l0-0.1_max-1.0_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.99_rho-0.05_l0-0.1_max-1.0_{}{}".format(metric, alpha),

            "fair_fate_b0-0.8_rho-0.04_l0-0.5_max-1.0_{}{}".format(metric, alpha),
            "fair_fate_b0-0.8_rho-0.05_l0-0.5_max-1.0_{}{}".format(metric, alpha),
            "fair_fate_b0-0.85_rho-0.04_l0-0.5_max-1.0_{}{}".format(metric, alpha),
            "fair_fate_b0-0.85_rho-0.05_l0-0.5_max-1.0_{}{}".format(metric, alpha),
            "fair_fate_b0-0.9_rho-0.04_l0-0.5_max-1.0_{}{}".format(metric, alpha),
            "fair_fate_b0-0.9_rho-0.05_l0-0.5_max-1.0_{}{}".format(metric, alpha),
            "fair_fate_b0-0.95_rho-0.04_l0-0.5_max-1.0_{}{}".format(metric, alpha),
            "fair_fate_b0-0.95_rho-0.05_l0-0.5_max-1.0_{}{}".format(metric, alpha),
            "fair_fate_b0-0.99_rho-0.04_l0-0.5_max-1.0_{}{}".format(metric, alpha),
            "fair_fate_b0-0.99_rho-0.05_l0-0.5_max-1.0_{}{}".format(metric, alpha),

            #"fair_fate_b0-0.8_rho-0.04_l0-0.1_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.8_rho-0.05_l0-0.1_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.85_rho-0.04_l0-0.1_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.85_rho-0.05_l0-0.1_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.9_rho-0.04_l0-0.1_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.9_rho-0.05_l0-0.1_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.95_rho-0.04_l0-0.1_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.95_rho-0.05_l0-0.1_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.99_rho-0.04_l0-0.1_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.99_rho-0.05_l0-0.1_max-0.9_{}{}".format(metric, alpha),

            #"fair_fate_b0-0.8_rho-0.04_l0-0.5_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.8_rho-0.05_l0-0.5_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.85_rho-0.04_l0-0.5_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.85_rho-0.05_l0-0.5_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.9_rho-0.04_l0-0.5_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.9_rho-0.05_l0-0.5_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.95_rho-0.04_l0-0.5_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.95_rho-0.05_l0-0.5_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.99_rho-0.04_l0-0.5_max-0.9_{}{}".format(metric, alpha),
            #"fair_fate_b0-0.99_rho-0.05_l0-0.5_max-0.9_{}{}".format(metric, alpha),

            # "fair_fate_b0-0.8_rho-0.04_l0-0.1_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.8_rho-0.05_l0-0.1_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.85_rho-0.04_l0-0.1_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.85_rho-0.05_l0-0.1_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.9_rho-0.04_l0-0.1_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.9_rho-0.05_l0-0.1_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.95_rho-0.04_l0-0.1_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.95_rho-0.05_l0-0.1_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.99_rho-0.04_l0-0.1_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.99_rho-0.05_l0-0.1_max-0.8_{}{}".format(metric, alpha),

            # "fair_fate_b0-0.8_rho-0.04_l0-0.5_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.8_rho-0.05_l0-0.5_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.85_rho-0.04_l0-0.5_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.85_rho-0.05_l0-0.5_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.9_rho-0.04_l0-0.5_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.9_rho-0.05_l0-0.5_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.95_rho-0.04_l0-0.5_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.95_rho-0.05_l0-0.5_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.99_rho-0.04_l0-0.5_max-0.8_{}{}".format(metric, alpha),
            # "fair_fate_b0-0.99_rho-0.05_l0-0.5_max-0.8_{}{}".format(metric, alpha),
        ]],
        [
            "{} {} - {} ".format(name, metric, alpha)
        ],
        [
            #"FedAvg",
            #"FedAvg+LR",
            "FedAvg+GR",

            #"FedMom b=0.8",
            #"FedMom b=0.85",
            #"FedMom b=0.9",
            #"FedMom b=0.95",
            #"FedMom b=0.99",

            #"FedDemon b0=0.8",
            #"FedDemon b0=0.85",
            #"FedDemon b0=0.9",
            #"FedDemon b0=0.95",
            #"FedDemon b0=0.99",

            #"FedVal (F={})".format(metric),

            #"FairFate (F={}) b0-0.8 rho-0.04 l0=0.1 max=1.0".format(metric),
            #"FairFate (F={}) b0-0.8 rho-0.05 l0=0.1 max=1.0".format(metric),
            #"FairFate (F={}) b0-0.85 rho-0.04 l0=0.1 max=1.0".format(metric),
            #"FairFate (F={}) b0-0.85 rho-0.05 l0=0.1 max=1.0".format(metric),
            #"FairFate (F={}) b0-0.9 rho-0.04 l0=0.1 max=1.0".format(metric),
            #"FairFate (F={}) b0-0.9 rho-0.05 l0=0.1 max=1.0".format(metric),
            #"FairFate (F={}) b0-0.95 rho-0.04 l0=0.1 max=1.0".format(metric),
            #"FairFate (F={}) b0-0.95 rho-0.05 l0=0.1 max=1.0".format(metric),
            #"FairFate (F={}) b0-0.99 rho-0.04 l0=0.1 max=1.0".format(metric),
            #"FairFate (F={}) b0-0.99 rho-0.05 l0=0.1 max=1.0".format(metric),

            "FairFate (F={}) b0-0.8 rho-0.04 l0=0.5 max=1.0".format(metric),
            "FairFate (F={}) b0-0.8 rho-0.05 l0=0.5 max=1.0".format(metric),
            "FairFate (F={}) b0-0.85 rho-0.04 l0=0.5 max=1.0".format(metric),
            "FairFate (F={}) b0-0.85 rho-0.05 l0=0.5 max=1.0".format(metric),
            "FairFate (F={}) b0-0.9 rho-0.04 l0=0.5 max=1.0".format(metric),
            "FairFate (F={}) b0-0.9 rho-0.05 l0=0.5 max=1.0".format(metric),
            "FairFate (F={}) b0-0.95 rho-0.04 l0=0.5 max=1.0".format(metric),
            "FairFate (F={}) b0-0.95 rho-0.05 l0=0.5 max=1.0".format(metric),
            "FairFate (F={}) b0-0.99 rho-0.04 l0=0.5 max=1.0".format(metric),
            "FairFate (F={}) b0-0.99 rho-0.05 l0=0.5 max=1.0".format(metric),

            #"FairFate (F={}) b0-0.8 rho-0.04 l0=0.1 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.8 rho-0.05 l0=0.1 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.85 rho-0.04 l0=0.1 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.85 rho-0.05 l0=0.1 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.9 rho-0.04 l0=0.1 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.9 rho-0.05 l0=0.1 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.95 rho-0.04 l0=0.1 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.95 rho-0.05 l0=0.1 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.99 rho-0.04 l0=0.1 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.99 rho-0.05 l0=0.1 max=0.9".format(metric),

            #"FairFate (F={}) b0-0.8 rho-0.04 l0=0.5 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.8 rho-0.05 l0=0.5 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.85 rho-0.04 l0=0.5 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.85 rho-0.05 l0=0.5 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.9 rho-0.04 l0=0.5 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.9 rho-0.05 l0=0.5 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.95 rho-0.04 l0=0.5 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.95 rho-0.05 l0=0.5 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.99 rho-0.04 l0=0.5 max=0.9".format(metric),
            #"FairFate (F={}) b0-0.99 rho-0.05 l0=0.5 max=0.9".format(metric),

            # "FairFate (F={}) b0-0.8 rho-0.04 l0=0.1 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.8 rho-0.05 l0=0.1 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.85 rho-0.04 l0=0.1 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.85 rho-0.05 l0=0.1 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.9 rho-0.04 l0=0.1 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.9 rho-0.05 l0=0.1 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.95 rho-0.04 l0=0.1 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.95 rho-0.05 l0=0.1 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.99 rho-0.04 l0=0.1 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.99 rho-0.05 l0=0.1 max=0.8".format(metric),

            # "FairFate (F={}) b0-0.8 rho-0.04 l0=0.5 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.8 rho-0.05 l0=0.5 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.85 rho-0.04 l0=0.5 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.85 rho-0.05 l0=0.5 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.9 rho-0.04 l0=0.5 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.9 rho-0.05 l0=0.5 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.95 rho-0.04 l0=0.5 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.95 rho-0.05 l0=0.5 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.99 rho-0.04 l0=0.5 max=0.8".format(metric),
            # "FairFate (F={}) b0-0.99 rho-0.05 l0=0.5 max=0.8".format(metric),
        ]
    )"""

    #plot_paretos(Adult().name, 20, 50, [None, 1.0, 0.5], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-adult")
    #plot_paretos(Compas().name, 20, 50, [None, 0.5, 0.25], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-compas")
    #plot_paretos(Dutch().name, 20, 50, [None, 1.0, 0.5], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-dutch")
    #plot_paretos(Law().name, 20, 50, [None, 0.5, 0.25], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-law")
