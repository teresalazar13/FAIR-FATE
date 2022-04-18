from code.plots.pareto_front import get_random_colors
from code.plots.plot import get_dfs

import matplotlib.pyplot as plt


def plot_results_epochs(dataset_name, num_runs, fairness_metrics, alphas, lambdas_fairfate, betas_fairfate, beta_fedmom):
    dfs = []
    metrics = ["ACC"]
    metrics.extend(fairness_metrics)

    for a in range(len(alphas)):
        fls = ["fedavg", "fedavg_gr", "fedavg_lr"]
        fairness_metrics_string = "-".join([f.split("_")[0] for f in fairness_metrics])
        fl_fedval = "fed_val_{}".format(fairness_metrics_string)
        fls.append(fl_fedval)
        fl_fairfed = "fair_fed_{}".format(fairness_metrics_string)
        fls.append(fl_fairfed)
        fls.append("fedmom_b_{}".format(str(beta_fedmom[a])))
        dfs_alpha = []

        for i in range(len(fairness_metrics)):
            fls_metric = fls[:]
            fairfate_fl = "fair_fate_l_e{}_b_{}_{}".format(str(lambdas_fairfate[a][i]), str(betas_fairfate[a][i]), fairness_metrics_string)
            fls_metric.append(fairfate_fl)
            if alphas[a]:
                for j in range(len(fls_metric)):
                    fls_metric[j] = "{}_alpha-{}".format(fls_metric[j], alphas[a])
            print(fls_metric)
            dfs_alpha.append(get_dfs(fls_metric, dataset_name, num_runs, metrics, fairness_metrics, True))
        dfs_alpha.append(dfs_alpha[0][0:4])
        dfs_alpha[-1].extend([dfs_alpha[0][-1], dfs_alpha[1][-1], dfs_alpha[2][-1]])
        dfs_alpha.insert(0, dfs_alpha.pop())
        dfs.extend(dfs_alpha)

    plt.figure(figsize=(23, 15))
    x_plot = [i for i in range(0, len(dfs[0][0]))]
    cols = 4
    rows = 3
    count = 1

    fls_legend = [
        ["FedAvg", "FedAvg+GR", "FedAvg+LR", "FedMom", "FAIR-FATE-SP", "FAIR-FATE-EO", "FAIR-FATE-EQO"],
        ["FedAvg", "FedAvg+GR", "FedAvg+LR", "FedMom", "FAIR-FATE-SP"],
        ["FedAvg", "FedAvg+GR", "FedAvg+LR", "FedMom", "FAIR-FATE-EO"],
        ["FedAvg", "FedAvg+GR", "FedAvg+LR", "FedMom", "FAIR-FATE-EQO"]
    ]
    colors = get_random_colors(len(fls_legend[0]))
    d = {}
    for i in range(len(fls_legend[0])):
        d[fls_legend[0][i]] = colors[i]

    for i in range(len(dfs)):
        i_ = i % len(metrics)
        alpha_title = r'$\alpha={}$'.format(alphas[i // len(metrics)])
        if not alphas[i // len(metrics)]:
            alpha_title = "RND"
        plt.subplot(rows, cols, count).set_title(alpha_title)
        for j in range(len(dfs[i])):
            plt.plot(x_plot, dfs[i][j][metrics[i_]].tolist(), color=d[fls_legend[i_][j]])
        plt.xlabel("Round Number")
        plt.ylabel(metrics[i_].replace("_ratio", "").replace("TPR", "EO"))
        plt.ylim([0, 1])
        count += 1

    handles = [plt.plot([], [], color=colors[i], marker="o", ls="")[0] for i in range(len(colors))]
    coords = (-1.8, -0.3)
    rho_legend = plt.legend(handles=handles, labels=fls_legend[0], loc=coords, prop={'size': 11}, ncol=len(handles))
    plt.gca().add_artist(rho_legend)

    plt.savefig('./datasets/{}/rounds_plot.png'.format(dataset_name), bbox_inches='tight')
    # plt.show()


"""
#plot_results_epochs(dataset_name, 20, metrics, alpha, lambda_fairfate, beta_fairfate, beta_fedmom)

def plot_results_epochs(dataset_name, num_runs, fairness_metrics, alpha, lambda_fairfate, beta_fairfate, beta_fedmom):
    fls = ["fedavg", "fedavg_gr", "fedavg_lr"]
    fairness_metrics_string = "-".join([f.split("_")[0] for f in fairness_metrics])
    fls.append("fedmom_b_{}".format(str(beta_fedmom)))
    fls.append("fair_fate_l_e{}_b_{}_{}".format(str(lambda_fairfate), str(beta_fairfate), fairness_metrics_string))
    metrics = ["ACC"]
    metrics.extend(fairness_metrics)

    if alpha:
        for i in range(len(fls)):
            fls[i] = "{}_alpha-{}".format(fls[i], alpha)

    dfs = get_dfs(fls, dataset_name, num_runs, metrics, fairness_metrics, True)

    plt.figure(figsize=(5, 90))
    x_plot = [i for i in range(0, len(dfs[0]))]
    cols = 1
    rows = len(dfs[0].columns)
    count = 1

    fls_legend = ["FedAvg", "FedAvg+GR", "FedAvg+LR", "FedMom", "FAIR-FATE"]
    for metric in metrics:
        plt.subplot(rows, cols, count)
        for df in dfs:
            plt.plot(x_plot, df[metric].tolist())
        plt.ylim([0, 1])
        plt.xlabel("Round Number")
        plt.ylabel(metric.replace("_ratio", "").replace("TPR", "EO"))
        plt.legend(fls_legend, loc="lower right")
        count += 1

    plt.savefig('./datasets/{}/rounds_plot_{}_alpha-{}.png'.format(dataset_name, fairness_metrics_string.replace("_ratio", ""), alpha), bbox_inches='tight')
    # plt.show()"""
