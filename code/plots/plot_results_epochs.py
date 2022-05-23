from code.plots.plot import get_dfs

import pandas as pd
import distinctipy
import matplotlib
import matplotlib.pyplot as plt


def plot_results_epochs(
        n_rounds, dataset_name, num_runs, fairness_metrics, alphas,
        betas_fairfate, rhos_fairfate, l0s_fairfate, maxs_fairfate,
        beta_fedmom, beta_fed_demon
):
    dfs = []
    metrics = ["ACC"]
    metrics.extend(fairness_metrics)

    for a in range(len(alphas)):
        fls = ["fedavg", "fedavg_gr", "fedavg_lr"]
        fls.append("fedmom_b_{}".format(str(beta_fedmom[a])))
        fls.append("fed_demon_b_{}".format(str(beta_fed_demon[a])))
        dfs_alpha = []

        for i in range(len(fairness_metrics)):
            fls_metric = fls[:]
            fairness_metrics_string = fairness_metrics[i].split("_")[0]
            fl_fedval = "fed_val_{}".format(fairness_metrics_string)
            fls_metric.append(fl_fedval)
            fairfate_fl = "fair_fate_b0-{}_rho-{}_l0-{}_max-{}_{}".format(
                str(betas_fairfate[a][i]), str(rhos_fairfate[a][i]), str(l0s_fairfate[a][i]), str(maxs_fairfate[a][i]),
                fairness_metrics_string
            )
            fls_metric.append(fairfate_fl)
            if alphas[a]:
                for j in range(len(fls_metric)):
                    fls_metric[j] = "{}_alpha-{}".format(fls_metric[j], alphas[a])
            print(fls_metric)
            dfs_alpha.append(get_dfs(n_rounds, fls_metric, dataset_name, num_runs, metrics, fairness_metrics, True))
        dfs_alpha.append(dfs_alpha[0][0:4])
        dfs_alpha[-1].extend([dfs_alpha[0][-1], dfs_alpha[1][-1], dfs_alpha[2][-1]])
        dfs_alpha.insert(0, dfs_alpha.pop())
        dfs.extend(dfs_alpha)

    plt.figure(figsize=(17, 14))
    x_plot = [i for i in range(0, len(dfs[0][0]))]
    fls_legend = [
        ["FedAvg", "FedAvg+GR", "FedAvg+LR", "FedMom", "FedDemon", "FedVal (F=SP)", "FedVal (F=EO)", "FedVal (F=EQO)", "FAIR-FATE (F=SP)", "FAIR-FATE (F=EO)", "FAIR-FATE (F=EQO)"],
        ["FedAvg", "FedAvg+GR", "FedAvg+LR", "FedMom", "FedDemon", "FedVal (F=SP)", "FAIR-FATE (F=SP)"],
        ["FedAvg", "FedAvg+GR", "FedAvg+LR", "FedMom", "FedDemon", "FedVal (F=EO)", "FAIR-FATE (F=EO)"],
        ["FedAvg", "FedAvg+GR", "FedAvg+LR", "FedMom", "FedDemon", "FedVal (F=EQO)", "FAIR-FATE (F=EQO)"]
    ]
    colors = distinctipy.get_colors(len(fls_legend[0]), rng=1)
    d = {}
    for i in range(len(fls_legend[0])):
        d[fls_legend[0][i]] = colors[i]
    for i in range(len(dfs)):
        i_ = i % len(metrics)
        alpha_title = r'$\alpha={}$'.format(alphas[i // len(metrics)])
        if not alphas[i // len(metrics)]:
            alpha_title = "RND"
        plt.subplot(3, 4, i + 1).set_title(alpha_title)
        for j in range(len(dfs[i])):
            plt.plot(x_plot, dfs[i][j][metrics[i_]].tolist(), color=d[fls_legend[i_][j]])
        plt.xlabel("Round Number")
        plt.ylabel(metrics[i_].replace("_ratio", "").replace("TPR", "EO"))
        plt.xlim([0, n_rounds])
        plt.ylim([0, 1])

    handles = [plt.plot([], [], color=colors[i], marker="o", ls="")[0] for i in range(len(colors))]
    coords = (-2.3, -0.64)
    rho_legend = plt.legend(handles=handles, labels=fls_legend[0], loc=coords, ncol=int(len(handles)/2))
    plt.gca().add_artist(rho_legend)
    plt.savefig('./datasets/{}/rounds_plot.png'.format(dataset_name), bbox_inches='tight')
    plt.show()


def plot_results_epochs_specific(dataset_name, num_runs, num_rounds, metric, fls_array, titles, fls_legend):
    colors = distinctipy.get_colors(len(fls_legend), rng=10)
    plt.figure(figsize=(7, 10))
    count = 0

    for fls in fls_array:
        dfs = []
        for fl in fls:
            df = get_avg_df_specific(fl, dataset_name, num_runs)
            dfs.append(df)
        x_plot = [i for i in range(0, num_rounds)]

        plt.subplot(len(fls_array), 2, count*2 + 2)
        i = 0
        for df in dfs:
            print(len(df["ACC"].tolist()))
            plt.plot(x_plot, df["ACC"].tolist()[:num_rounds], color=colors[i])
            plt.xlabel("Round Number")
            plt.ylabel("ACC")
            i += 1
        plt.ylim([0, 1])
        plt.title(titles[count])
        plt.subplot(len(fls_array), 2, count*2 + 1)
        i = 0
        for df in dfs:
            plt.plot(x_plot, df["{}_ratio".format(metric)].tolist()[:num_rounds], color=colors[i])
            plt.xlabel("Round Number")
            plt.ylabel(metric)
            i += 1
        plt.ylim([0, 1.2])
        plt.title(titles[count])
        count += 1

    coords = (0.42, -0.7)
    handles = [plt.plot([], [], color=colors[i], marker="o", ls="")[0] for i in range(len(colors))]
    legend = plt.legend(handles=handles, labels=fls_legend, loc=coords, ncol=len(fls_legend) // 2)
    plt.gca().add_artist(legend)
    plt.tight_layout()
    plt.show()


def get_avg_df_specific(fl, dataset_name, num_runs):
    dfs = []
    for run_num in range(1, num_runs + 1):
        filename = './datasets/{}/run_{}/{}.csv'.format(dataset_name, run_num, fl)
        df = pd.read_csv(filename)
        dfs.append(df)

    df_concat = pd.concat((dfs))
    df_concat = df_concat.groupby(df_concat.index)
    df_concat_avg = df_concat.mean()

    return df_concat_avg
