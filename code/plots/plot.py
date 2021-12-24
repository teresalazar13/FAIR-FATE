import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from kneed import KneeLocator


def plot_avg_results(dataset_name, num_runs):
    fls = ["fedavg_alpha-2", "fedavg_gr_alpha-2", "fedavg_lr_alpha-2"]
    metrics = ["TPR_ratio"]
    metrics_results = ["ACC", "F1Score", "MCC", "TPR_ratio"]
    fedavg_acc = get_avg_df(fls[0], dataset_name, num_runs, metrics_results)["ACC"].iloc[-1]
    best = [0 for _ in metrics]
    dfs = []

    for fl in fls:
        df = get_avg_df(fl, dataset_name, num_runs, metrics_results)
        dfs.append(df)
        for i in range(len(metrics)):
            value = df[metrics[i]].iloc[-1]
            if value > best[i]:
                best[i] = value

    # FAIR-FATE exponential
    fls_fair_fate_exp = []
    dfs_fair_fate_exp = []
    for beta in [0.7, 0.8, 0.9, 0.99]:
        for lambda_exponential in [0.04, 0.045, 0.05]:
            fl = "fair_fate_l_e{}_b_{}_TPR_alpha-2".format(str(lambda_exponential), str(beta))
            fls_fair_fate_exp.append(fl)
            df = get_avg_df(fl, dataset_name, num_runs, metrics_results)
            dfs_fair_fate_exp.append(df)
    best_df_fair_fate, best_fl_fair_fate = get_best_fl_group(fls_fair_fate_exp, dfs_fair_fate_exp, metrics, best, fedavg_acc)
    dfs.append(best_df_fair_fate)
    fls.append(best_fl_fair_fate)

    #fl = "fair_fate_l_e0.05_b_0.9_TPR"
    #fls.append(fl)
    #df = get_avg_df(fl, dataset_name, num_runs, metrics_results)
    #dfs.append(df)

    # FedMom
    fls_fedmom = []
    dfs_fedmom = []
    for beta in [0.7, 0.8, 0.9, 0.99]:
        fl = "fedmom_b_{}_alpha-2".format(str(beta))
        fls_fedmom.append(fl)
        df = get_avg_df(fl, dataset_name, num_runs, metrics_results)
        dfs_fedmom.append(df)
    best_df_fedmom, best_fl_fedmom = get_best_fl_group(fls_fedmom, dfs_fedmom, metrics, best, fedavg_acc)
    dfs.append(best_df_fedmom)
    fls.append(best_fl_fedmom)

    plot_results(dfs, fls, './datasets/{}/rounds_plot_alpha-1.png'.format(dataset_name), metrics_results)
    get_last_round_plot(dfs, fls, './datasets/{}/last_round_plot_alpha-1.png'.format(dataset_name), metrics_results)
    plot_pareto_front(dfs_fair_fate_exp, fls_fair_fate_exp, './datasets/{}/pareto_front_alpha-1.png'.format(dataset_name), "ACC", "TPR_ratio")


def plot_pareto_front(dfs, fls, filename, metric_a, metric_b):
    x = []
    y = []
    labels = []
    costs = []

    for i in range(len(dfs)):
        if "b_0.7_" in fls[i]:
            print(fls[i])
            labels.append(fls[i])
            value_a = dfs[i][metric_a].iloc[-1]
            value_b = dfs[i][metric_b].iloc[-1]
            x.append(value_a)
            y.append(value_b)
            costs.append([x, y])

    #kneedle = KneeLocator(x, y, curve="convex", direction="increasing")
    #print(kneedle.knee)
    plt.figure(figsize=(5, 5))
    plt.xlabel(metric_a)
    plt.ylabel(metric_b)
    plt.scatter(x, y)

    for i in range(len(x)):
        plt.annotate(labels[i], (x[i], y[i]))

    plt.savefig(filename, bbox_inches='tight')
    # plt.show()


def get_avg_df(name, dataset_name, num_runs, metric_results):
    dfs = []
    for run_num in range(1, num_runs + 1):
        filename = './datasets/{}/run_{}/{}.csv'.format(dataset_name, run_num, name)
        df = pd.read_csv(filename)
        dfs.append(df)

    df_concat = pd.concat((dfs))
    df_concat = df_concat.groupby(df_concat.index)
    df_concat_avg = df_concat.mean()
    df_concat_std = df_concat.std()
    for metric in metric_results:
        print("{} - {}: {}+-{}".format(name, metric, round(df_concat_avg[metric].iloc[-1], 2), round(df_concat_std[metric].iloc[-1], 2)))

    return df_concat_avg


def plot_results(dfs, fls, plot_filename, metrics):
    plt.figure(figsize=(5, 90))
    x_plot = [i for i in range(0, len(dfs[0]))]
    cols = 1
    rows = len(dfs[0].columns)
    count = 1

    for metric in metrics:
        plt.subplot(rows, cols, count)
        for df in dfs:
            plt.plot(x_plot, df[metric].tolist())
        plt.ylim([0, 1])
        plt.xlabel("Round Number")
        plt.ylabel(metric)
        plt.legend(fls, loc="lower right")
        count += 1

    plt.savefig(plot_filename, bbox_inches='tight')
    # plt.show()


def get_last_round_plot(dfs, fls, plot_filename, metrics):
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    space = 0.5 / len(dfs)
    br = np.arange(len(metrics))

    for i in range(len(dfs)):
        values = [dfs[i][metric].iloc[-1] for metric in metrics]
        ax.bar(br, values, width=space, label=fls[i])
        br = [x + space for x in br]

    plt.xticks([r + space for r in range(len(metrics))], metrics)
    plt.legend()
    plt.savefig(plot_filename, bbox_inches='tight')
    # plt.show()


def get_best_fl_group(fls, dfs, metrics, best, fedavg_acc):
    improvs_fair_fate = []

    for i in range(len(dfs)):
        df = dfs[i]
        values = [df[metric].iloc[-1] for metric in metrics]
        improv = 0
        acc = df["ACC"].iloc[-1]

        for i in range(len(metrics)):
            value = df[metrics[i]].iloc[-1]
            improv += value / best[i] - 1
        improvs_fair_fate.append(improv)
        # print(fls[i])
        # print(values)
        # print(round(improv, 2))
    max_idx = np.argmax(improvs_fair_fate)

    return dfs[max_idx], fls[max_idx]
