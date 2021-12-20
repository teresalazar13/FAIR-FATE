import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_avg_results(dataset_name, num_runs):
    fls = ["fedavg", "fedavg_gr", "fedavg_lr"]
    metrics = ["TPR_ratio"]
    fedavg_acc = get_metrics_fd(fls[0], dataset_name, num_runs)["ACC"].iloc[-1]
    best = [0 for _ in metrics]
    dfs = []

    for fl in fls:
        df = get_metrics_fd(fl, dataset_name, num_runs)
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
            fl = "fair_fate_l_e{}_b_{}_TPR".format(str(lambda_exponential), str(beta))
            fls_fair_fate_exp.append(fl)
            df = get_metrics_fd(fl, dataset_name, num_runs)
            dfs_fair_fate_exp.append(df)
    best_df_fair_fate, best_fl_fair_fate = get_best_fl_group(fls_fair_fate_exp, dfs_fair_fate_exp, dataset_name, num_runs, metrics, best, fedavg_acc)
    dfs.append(best_df_fair_fate)
    fls.append(best_fl_fair_fate)

    # FAIR-FATE fixed
    fls_fair_fate_fixed = []
    dfs_fair_fate_fixed = []
    for beta in [0.7, 0.8, 0.9, 0.99]:
        for lambda_fixed in [0.5, 0.6, 0.7, 0.8, 0.9]:
            fl = "fair_fate_l_f{}_b_{}_TPR".format(str(lambda_fixed), str(beta))
            fls_fair_fate_fixed.append(fl)
            df = get_metrics_fd(fl, dataset_name, num_runs)
            dfs_fair_fate_fixed.append(df)
    best_df_fair_fate, best_fl_fair_fate = get_best_fl_group(fls_fair_fate_fixed, dfs_fair_fate_fixed, dataset_name, num_runs, metrics, best, fedavg_acc)
    dfs.append(best_df_fair_fate)
    fls.append(best_fl_fair_fate)

    # FedMom
    fls_fedmom = []
    dfs_fedmom = []
    for beta in [0.7, 0.8, 0.9, 0.99]:
        fl = "fedmom_b_{}".format(str(beta))
        fls_fedmom.append(fl)
        df = get_metrics_fd(fl, dataset_name, num_runs)
        dfs_fedmom.append(df)
    best_df_fedmom, best_fl_fedmom = get_best_fl_group(fls_fedmom, dfs_fedmom, dataset_name, num_runs, metrics, best, fedavg_acc)
    dfs.append(best_df_fedmom)
    fls.append(best_fl_fedmom)

    #metrics_results = ["ACC", "F1Score", "MCC", "TPR_ratio"]
    #plot_results(dfs, fls, './datasets/{}/rounds_plot.png'.format(dataset_name), metrics_results)
    #get_last_round_plot(dfs, fls, './datasets/{}/last_round_plot.png'.format(dataset_name), metrics_results)
    plot_pareto_front(dfs_fair_fate_fixed, fls_fair_fate_fixed, "ACC", "TPR_ratio")


def plot_pareto_front(dfs, fls, metric_a, metric_b):
    x = []
    y = []

    for i in range(len(dfs)):
        value_a = dfs[i][metric_a].iloc[-1]
        value_b = dfs[i][metric_b].iloc[-1]
        x.append(value_a)
        y.append(value_b)

    plt.xlabel(metric_a)
    plt.ylabel(metric_b)
    plt.scatter(x, y)
    for i in range(len(fls)):
        plt.annotate(fls[i], (x[i], y[i]))

    plt.show()


def get_metrics_fd(name, dataset_name, num_runs):
    dfs = []
    for run_num in range(1, num_runs + 1):
        filename = './datasets/{}/run_{}/{}.csv'.format(dataset_name, run_num, name)
        df = pd.read_csv(filename)
        dfs.append(df)

    df_concat = pd.concat((dfs))
    df_concat = df_concat.groupby(df_concat.index)
    df_concat = df_concat.mean()

    return df_concat


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
    count = 0

    for df in dfs:
        values = [df[metric].iloc[-1] for metric in metrics]
        ax.bar(br, values, width=space, label=fls[count])
        br = [x + space for x in br]
        count += 1

    plt.xticks([r + space for r in range(len(metrics))], metrics)
    plt.legend()
    plt.savefig(plot_filename, bbox_inches='tight')
    # plt.show()


def get_best_fl_group(fls, dfs, dataset_name, num_runs, metrics, best, fedavg_acc):
    dfs_fair_fate = []
    improvs_fair_fate = []

    for fl in fls:
        df = get_metrics_fd(fl, dataset_name, num_runs)
        dfs_fair_fate.append(df)
        values = [df[metric].iloc[-1] for metric in metrics]
        improv = 0
        acc = df["ACC"].iloc[-1]

        for i in range(len(metrics)):
            value = df[metrics[i]].iloc[-1]
            improv += value / best[i] - 1
        improvs_fair_fate.append(improv)
        print(fl)
        # print(values)
        print(round(improv, 2))
    max_idx = np.argmax(improvs_fair_fate)

    return dfs_fair_fate[max_idx], fls[max_idx]
