from code.plots.plot import plot_avg_results


def plot(dataset_name, num_runs, fairness_metrics, alpha=None, results_folder=""):
    metrics_results = ["ACC", "F1Score", "MCC"]
    metrics_results.extend(fairness_metrics)
    fls = ["fedavg", "fedavg_gr", "fedavg_lr"]

    fls_fair_fate = []
    for beta in [0.7, 0.8, 0.9, 0.99]:
        for lambda_exponential in [0.04, 0.045, 0.05]:
            fl = "fair_fate_l_e{}_b_{}_TPR".format(str(lambda_exponential), str(beta))
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

    plot_avg_results(dataset_name, num_runs, fls, fls_fair_fate, fls_fedmom, fairness_metrics, metrics_results, results_folder, alpha)
