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
    alpha = 1.0
    dataset_name = Adult().name
    metrics = ["TPR_ratio"]
    #print_results(dataset_name, 20, 50, metrics, alpha=alpha)

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
    """
    plot_results_epochs_specific(
        Law().name, 20,
        [[
            "fedavg_alpha-0.25",
            "fedmom_b_0.8_alpha-0.25",
            "fedavg_gr_alpha-0.25",
            "fedavg_lr_alpha-0.25",
            "fed_val_TPR_alpha-0.25",
            "fair_fate_rho-0.04_l0-0.1_max-1.0_b-0.7_TPR_alpha-0.25"
        ],
        [
            "fedavg_alpha-0.5",
            "fedmom_b_0.7_alpha-0.5",
            "fedavg_gr_alpha-0.5",
            "fedavg_lr_alpha-0.5",
            "fed_val_TPR_alpha-0.5",
            "fair_fate_rho-0.047_l0-0.1_max-1.0_b-0.9_TPR_alpha-0.5"
        ],
        [
            "fedavg",
            "fedmom_b_0.99",
            "fedavg_gr",
            "fedavg_lr",
            "fed_val_TPR",
            "fair_fate_rho-0.04_l0-0.1_max-1.0_b-0.8_TPR"
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

    """
    plot_results_epochs_specific(
        Compas().name, 5, 100,
        [[
            #"fedavg_alpha-0.25",
            #"fedavg_gr_alpha-0.25",
            "fedavg_lr_alpha-0.25",
            #"fair_fate_rho-0.045_l0-1.0_max-10000_b-0.9_SP_alpha-0.25",
            "fair_fate_rho-0.045_l0-0.1_max-1.0_b-0.9_SP_alpha-0.25",
            #"fair_fate_rho-0.03_l0-0.1_max-1.0_b-0.9_SP_alpha-0.25",
            #"fair_fate_rho-0.03_l0-0.1_max-1.0_b-0.7_SP_alpha-0.25",
            "fair_fate_rho-0.045_l0-0.1_max-0.6_b-0.9_SP_alpha-0.25",
        ]],
        [
            r'$\alpha=0.25$'
        ],
        [
            #"FedAvg",
            #"FedAvg+GR",
            "FedAvg+LR",
            #'FAIR-FATE (F=SP) LOL',
            'FAIR-FATE (F=SP) the one chosen',
            #'FAIR-FATE (F=SP) less rho',
            #'FAIR-FATE (F=SP) less rho less beta with decay',
            'FAIR-FATE (F=SP) lower max and weight decay',
        ]
    )"""

    """
    # Adult SP RANDOM comparison com weight decay
    plot_results_epochs_specific(
        Adult().name, 1, 100, "SP"
        [[
            "fedavg_lr",
            "fair_fate_rho-0.05_l0-0.1_max-1.0_b-0.99_SP",
            #"fair_fate_rho-0.035_l0-0.1_max-1.0_b-0.99_SP",
            #"fair_fate_rho-0.05_l0-0.1_max-1.0_b-0.7_SP",
            #"fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.99_SP",
            #"fair_fate_decay_rho-0.06_l0-0.1_max-1.0_b-0.9_SP",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.95_SP",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.975_SP",
        ]],
        [
            "RND"
        ],
        [
            "FedAvg+LR",
            'FAIR-FATE (F=SP) the one chosen',
            #'FAIR-FATE (F=SP) lower lambda',
            #'FAIR-FATE (F=SP) lower beta',
            #'FAIR-FATE (F=SP) the one chosen + weight decay',
            #'FAIR-FATE (F=SP) higher rho beta=0.9 + weight decay',
            'FAIR-FATE (F=SP) beta=0.95 + weight decay',
            'FAIR-FATE (F=SP) beta=0.975 + weight decay',
        ]
    )
    # Adult TPR RANDOM comparison com weight decay
    plot_results_epochs_specific(
        Adult().name, 1, 100, "TPR",
        [[
            "fedavg_lr",
            "fair_fate_rho-0.047_l0-0.1_max-1.0_b-0.99_TPR",
            #"fair_fate_decay_rho-0.047_l0-0.1_max-1.0_b-0.975_TPR",
            #"fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.975_TPR",
            #"fair_fate_decay_rho-0.06_l0-0.1_max-1.0_b-0.975_TPR",
            #"fair_fate_decay_rho-0.045_l0-0.1_max-1.0_b-0.975_TPR"
            #"fair_fate_decay_rho-0.047_l0-0.1_max-1.0_b-0.95_TPR",
            #"fair_fate_decay_rho-0.047_l0-0.1_max-1.0_b-0.99_TPR",
            #"fair_fate_decay_rho-0.047_l0-0.1_max-1.0_b-0.925_TPR",
            #"fair_fate_decay_rho-0.047_l0-0.1_max-1.0_b-0.93_TPR",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.93_TPR",
        ]],
        [
            "RND"
        ],
        [
            "FedAvg+LR",
            'FAIR-FATE (F=EO) the one chosen',
            #'FAIR-FATE (F=EO) beta=0.975 + decay',
            #'FAIR-FATE (F=EO) beta=0.975, rho=0.05 + decay',
            #'FAIR-FATE (F=EO) beta=0.975, rho=0.06 + decay',
            #'FAIR-FATE (F=EO) beta=0.975, rho=0.045 + decay'
            #'FAIR-FATE (F=EO) beta=0.95 + decay',
            #'FAIR-FATE (F=EO) beta=0.99 + decay',
            #'FAIR-FATE (F=EO) beta=0.925 + decay',
            #'FAIR-FATE (F=EO) beta=0.93 + decay',
            'FAIR-FATE (F=EO) beta=0.93, rho=0.05 + decay',
        ]
    )
    # Adult EQO RANDOM comparison com weight decay
    plot_results_epochs_specific(
        Adult().name, 1, 100, "EQO",
        [[
            "fedavg_lr",
            "fair_fate_rho-0.05_l0-0.1_max-1.0_b-0.99_EQO",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.95_EQO",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.975_EQO",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.93_EQO"
        ]],
        [
            "RND"
        ],
        [
            "FedAvg+LR",
            'FAIR-FATE (F=EQO) the one chosen',
            'FAIR-FATE (F=EQO) beta=0.95 decay',
            'FAIR-FATE (F=EQO) beta=0.975 decay',
            'FAIR-FATE (F=EQO) beta=0.93 decay'
        ]
    )
    # COMPAS SP RANDOM comparison com weight decay
    plot_results_epochs_specific(
        Compas().name, 1, 100, "SP",
        [[
            "fedavg_lr",
            "fair_fate_rho-0.05_l0-0.1_max-1.0_b-0.99_SP",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.93_SP",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.95_SP",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.975_SP"
        ]],
        [
            "RND"
        ],
        [
            "FedAvg+LR",
            'FAIR-FATE (F=SP) the one chosen',
            'FAIR-FATE (F=SP) beta=0.93 decay',
            'FAIR-FATE (F=SP) beta=0.95 decay',
            'FAIR-FATE (F=SP) beta=0.975 decay'
        ]
    )
    # COMPAS SP 0.25 comparison com weight decay
    plot_results_epochs_specific(
        Compas().name, 1, 100, "SP",
        [[
            "fed_val_SP_alpha-0.25",
            "fair_fate_rho-0.045_l0-0.1_max-1.0_b-0.9_SP_alpha-0.25",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.93_SP_alpha-0.25",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.95_SP_alpha-0.25",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.975_SP_alpha-0.25"
        ]],
        [
            "alpha=0.25"
        ],
        [
            "FedVal (F=SP)",
            'FAIR-FATE (F=SP) the one chosen',
            'FAIR-FATE (F=SP) beta=0.93 decay',
            'FAIR-FATE (F=SP) beta=0.95 decay',
            'FAIR-FATE (F=SP) beta=0.975 decay'
        ]
    )
    # DUTCH TPR 1.0 comparison com weight decay
    plot_results_epochs_specific(
        Dutch().name, 1, 100, "TPR",
        [[
            #"fedavg_gr_alpha-1.0",
            "fair_fate_rho-0.047_l0-0.1_max-1.0_b-0.8_TPR_alpha-1.0",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.93_TPR_alpha-1.0",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.95_TPR_alpha-1.0",
            "fair_fate_decay_rho-0.05_l0-0.1_max-1.0_b-0.975_TPR_alpha-1.0",
            "fair_fate_decay_rho-0.04_l0-0.1_max-1.0_b-0.95_TPR_alpha-1.0"
        ]],
        [
            "alpha=1.0"
        ],
        [
            #"FedAvg+LR",
            'FAIR-FATE (F=TPR) the one chosen',
            'FAIR-FATE (F=TPR) beta=0.93 decay',
            'FAIR-FATE (F=TPR) beta=0.95 decay',
            'FAIR-FATE (F=TPR) beta=0.975 decay',
            'FAIR-FATE (F=TPR) beta=0.95 0.4 rho decay',
        ]
    )"""

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
    )

    #plot_paretos(Adult().name, 20, 50, [None, 1.0, 0.5], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-adult")
    #plot_paretos(Compas().name, 20, 50, [None, 0.5, 0.25], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-compas")
    #plot_paretos(Dutch().name, 20, 50, [None, 1.0, 0.5], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-dutch")
    #plot_paretos(Law().name, 20, 50, [None, 0.5, 0.25], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-law")
