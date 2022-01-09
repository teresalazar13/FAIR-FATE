from code.plots.plot import plot_avg_results
from code.datasets.Adult import Adult
from code.datasets.Compas import Compas


def plot(dataset_name, num_runs, fairness_metrics, alpha=None):
    metrics_results = ["ACC", "F1Score", "MCC", "SP_ratio", "TPR_ratio", "EQO_ratio"]
    metrics_results.extend(fairness_metrics)
    fls = ["fedavg", "fedavg_gr", "fedavg_lr"]
    fairness_metrics_string = "-".join([f.split("_")[0] for f in fairness_metrics])

    fls_fair_fate = []
    for beta in [0.7, 0.8, 0.9, 0.99]:
        for lambda_init in [0.035, 0.04, 0.045, 0.047, 0.05]:
            fl = "fair_fate_l_e{}_b_{}_{}".format(str(lambda_init), str(beta), fairness_metrics_string)
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

    plot_avg_results(dataset_name, num_runs, fls, fls_fair_fate, fls_fedmom, fairness_metrics, metrics_results)


if __name__ == '__main__':
    plot(Compas().name, 10, ["EQO_ratio"])
