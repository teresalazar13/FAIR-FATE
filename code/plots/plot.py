import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_results(dfs, fls, plot_filename, metrics_type):
    plt.figure(figsize=(5, 90))
    x_plot = [i for i in range(0, len(dfs[0]))]
    cols = 1
    rows = len(dfs[0].columns)
    count = 1

    if isinstance(metrics_type, list):
        metrics = metrics_type
    elif metrics_type == "diff":
        metrics = [m for m in dfs[0].columns if "diff" in m or m == "ACC"]
    else:
        metrics = [m for m in dfs[0].columns if "diff" not in m]

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


def get_last_round_plot(dfs, fls, plot_filename, metrics_type):
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_axes([0, 0, 1, 1])

    if isinstance(metrics_type, list):
        metrics = metrics_type
    elif metrics_type == "diff":
        metrics = [m for m in dfs[0].columns if "diff" in m or m == "ACC"]
    else:
        metrics = [m for m in dfs[0].columns if "diff" not in m]
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


def plot_avg_results(dataset_name, num_runs):
    fls = ["fedavg", "fedavg_gr", "fedavg_lr"]
    dfs = []
    metrics = ["TPR_ratio"]
    fedavg_acc = get_metrics_fd(fls[0], dataset_name, num_runs)["ACC"].iloc[-1]
    best = [0 for _ in metrics]
    for fl in fls:
        df = get_metrics_fd(fl, dataset_name, num_runs)
        dfs.append(df)
        for i in range(len(metrics)):
            value = df[metrics[i]].iloc[-1]
            if value > best[i]:
                best[i] = value

    # FAIR-FATE
    fls_fair_fate = []
    for beta in [0.8, 0.9, 0.99]:
        for lambda_ in [0.045, 0.05, 0.07, 0.08]:
            fls_fair_fate.append("fair_fate_l_e{}_b_{}_TPR".format(str(lambda_), str(beta)))
    best_df_fair_fate, best_fl_fair_fate = get_best_fl_group(fls_fair_fate, dataset_name, num_runs, metrics, best, fedavg_acc)
    dfs.append(best_df_fair_fate)
    fls.append(best_fl_fair_fate)

    fl = "fair_fate_l_f0.5_b_0.9_TPR"
    df = get_metrics_fd(fl, dataset_name, num_runs)
    fls.append(fl)
    dfs.append(df)

    # FedMom
    fls_fedmom = []
    for beta in [0.8, 0.9, 0.99]:
        fls_fedmom.append("fedmom_b_{}".format(str(beta)))
    best_df_fedmom, best_fl_fedmom = get_best_fl_group(fls_fedmom, dataset_name, num_runs, metrics, best, fedavg_acc)
    dfs.append(best_df_fedmom)
    fls.append(best_fl_fedmom)

    plot_results(dfs, fls, './datasets/{}/rounds_plot.png'.format(dataset_name),
                 ["ACC", "F1Score", "MCC", "TPR_ratio"])
    get_last_round_plot(dfs, fls, './datasets/{}/last_round_plot.png'.format(dataset_name),
                        ["ACC", "F1Score", "MCC", "TPR_ratio"])
    # get_last_round_plot(dfs, fls, './datasets/{}/last_round_plot_diff.png'.format(dataset_name), "diff")
    # get_last_round_plot(dfs, fls, './datasets/{}/last_round_plot_ratio.png'.format(dataset_name), "ratio")


def get_best_fl_group(fls_fair_fate, dataset_name, num_runs, metrics, best, fedavg_acc):
    dfs_fair_fate = []
    improvs_fair_fate = []

    for fl in fls_fair_fate:
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
        #print(values)
        print(round(improv, 2))
    max_idx = np.argmax(improvs_fair_fate)

    return dfs_fair_fate[max_idx], fls_fair_fate[max_idx]
