from code.plots.pareto_front import plot_pareto_fronts
from code.plots.plot import plot_avg_results
from code.datasets.Adult import Adult
from code.datasets.Compas import Compas
from code.datasets.Law import Law
from code.datasets.Dutch import Dutch


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


def plot_paretos(dataset_name, num_runs, alphas, metrics_F, metric_a, filename):
    lambdas_ = [0.035, 0.04, 0.045, 0.047, 0.05]
    betas = [0.7, 0.8, 0.9, 0.99]
    fls_fair_fate = []

    for alpha in alphas:
        for metric_F in metrics_F:
            fls_fair_fate_alpha_metric = []
            for beta in betas:
                for lambda_init in lambdas_:
                    fl = "fair_fate_l_e{}_b_{}_{}".format(str(lambda_init), str(beta), metric_F[0].replace("_ratio", ""))
                    if alpha:
                        fl = "{}_alpha-{}".format(fl, alpha)
                    fls_fair_fate_alpha_metric.append(fl)
            fls_fair_fate.append([alpha, metric_F, fls_fair_fate_alpha_metric])

    plot_pareto_fronts(dataset_name, num_runs, fls_fair_fate, metric_a, lambdas_, betas, filename)


if __name__ == '__main__':
    plot(Dutch().name, 20, ["EQO_ratio"], alpha=0.5)
    #plot_paretos(Adult().name, 10, [None, 1.0, 0.5], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-adult")
    #plot_paretos(Compas().name, 10, [None, 0.5, 0.25], [["SP_ratio"], ["TPR_ratio"], ["EQO_ratio"]], "ACC", "pareto_fronts-compas")
