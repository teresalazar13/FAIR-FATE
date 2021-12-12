import matplotlib as plt
import pandas as pd
import numpy as np


def plot_results(dfs, fls, plot_filename):
    fig = plt.figure(figsize=(30, 120))
    x_plot = [i for i in range(0, len(dfs[0]))]
    cols = 1
    rows = len(dfs[0].columns)
    count = 1

    for metric in dfs[0].columns:
        plt.subplot(rows, cols, count)
        for df in dfs:
            plt.plot(x_plot, df[metric].tolist())

        plt.ylim([0, 1])
        plt.title(metric)
        plt.xlabel("Round Number")
        plt.ylabel(metric)
        plt.legend(fls, loc="lower right")
        count += 1

    plt.subplot_tool()
    plt.savefig(plot_filename)
    plt.show()


def get_last_round_plot(dfs, fls, plot_filename, type_):
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_axes([0, 0, 1, 1])

    if type_ == "diff":
        cols = [m for m in dfs[0].columns if "diff" in m or m == "ACC"]
    else:
        cols = [m for m in dfs[0].columns if "diff" not in m]

    br = np.arange(len(cols))
    count = 0
    for df in dfs:
        values = [df[metric].iloc[-1] for metric in cols]
        ax.bar(br, values, width=0.25, label=fls[count])
        br = [x + 0.25 for x in br]
        count += 1

    plt.xticks([r + 0.25 for r in range(len(cols))], cols)
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()


def get_metrics_fd(name, dataset_name, num_runs):
    dfs = []
    for run_num in range(num_runs):
        filename = './datasets/{}/run_{}/{}.csv'.format(dataset_name, run_num + 1, name)
        df = pd.read_csv(filename)
        dfs.append(df)

    df_concat = pd.concat((dfs))
    df_concat = df_concat.groupby(df_concat.index)
    df_concat = df_concat.mean()

    return df_concat


def plot_avg_results(dataset_name, num_runs):
    fls = ["fedavg", "fedavg_gr", "fedavg_lr", "fedavg_fair_acc_ratio"]

    """
    # FedMom
    for beta in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
      fls.append("fedmom_b_{}".format(str(beta)))"""

    """
    # FAIR-FATE
    for beta in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
      for lambda_ in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        fls.append("fair_fate_l_{}_b_{}".format(str(lambda_), str(beta)))"""

    dfs = []
    for fl in fls:
        df = get_metrics_fd(fl, dataset_name, num_runs)
        dfs.append(df)

    plot_results(dfs, fls, './datasets/{}/rounds_plot.png'.format(dataset_name))
    get_last_round_plot(dfs, fls, './datasets/{}/last_round_plot_diff.png'.format(dataset_name), "diff")
    get_last_round_plot(dfs, fls, './datasets/{}/last_round_plot_ratio.png'.format(dataset_name), "ratio")
