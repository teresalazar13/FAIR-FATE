import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_avg_results(dataset_name, num_runs, fls, fls_fair_fate, fls_fedmom, fairness_metrics, metrics_results, alpha):
    fedavg_acc = get_avg_df(fls[0], dataset_name, num_runs, metrics_results, fairness_metrics, True)["ACC"].iloc[-1]
    dfs = get_dfs(fls, dataset_name, num_runs, metrics_results, fairness_metrics, True)

    # FedMom
    dfs_fedmom = get_dfs(fls_fedmom, dataset_name, num_runs, metrics_results, fairness_metrics, True)
    fairness_metrics_all = ["SP_ratio", "TPR_ratio", "EQO_ratio"]
    best_df_fedmom, best_fl_fedmom = get_best_fl(dfs, fls_fedmom, dfs_fedmom, fairness_metrics_all, fedavg_acc)
    dfs.append(best_df_fedmom)
    fls.append(best_fl_fedmom)

    # FAIR-FATE
    dfs_fair_fate = get_dfs(fls_fair_fate, dataset_name, num_runs, metrics_results, fairness_metrics, False)
    best_df_fair_fate, best_fl_fair_fate = get_best_fl(dfs, fls_fair_fate, dfs_fair_fate, fairness_metrics, fedavg_acc)
    dfs.append(best_df_fair_fate)
    fls.append(best_fl_fair_fate)

    #plot_results(dfs, fls, './datasets/{}/rounds_plot.png'.format(dataset_name), metrics_results)
    #get_last_round_plot(dfs, fls, './datasets/{}/last_round_plot.png'.format(dataset_name), metrics_results)

def get_dfs(fls, dataset_name, num_runs, metrics_results, fairness_metrics, is_baseline):
    dfs = []
    for fl in fls:
        df = get_avg_df(fl, dataset_name, num_runs, metrics_results, fairness_metrics, is_baseline)
        dfs.append(df)

    return dfs


def get_best_fl(dfs, fls_alg, dfs_alg, fairness_metrics_all, fedavg_acc):
    best = [0 for _ in fairness_metrics_all]

    for df in dfs:
        for i in range(len(fairness_metrics_all)):
            value = df[fairness_metrics_all[i]].iloc[-1]
            if value > best[i]:
                best[i] = value
    best_df, best_fl = get_best_fl_group(fls_alg, dfs_alg, fairness_metrics_all, best, fedavg_acc)
    print("best - {}".format(best_fl))

    return best_df, best_fl


def get_best_fl_group(fls, dfs, metrics, best, fedavg_acc):
    improvs_fair_fate = []

    for i in range(len(dfs)):
        df = dfs[i]
        values = [df[metric].iloc[-1] for metric in metrics]
        improv = 0
        acc = df["ACC"].iloc[-1]

        for j in range(len(metrics)):
            value = df[metrics[j]].iloc[-1]
            improv += value / best[j] - 1
        improvs_fair_fate.append(improv)
        # print(fls[i])
        # print(values)
        print(round(improv, 2))
    max_idx = np.argmax(improvs_fair_fate)

    return dfs[max_idx], fls[max_idx]

def get_avg_df(fl, dataset_name, num_runs, metrics_results, fairness_metrics, is_baseline):
    dfs = []
    print(fl)
    print(fairness_metrics)
    for run_num in range(1, num_runs + 1):
        filename = get_filename(dataset_name, run_num, fl, is_baseline)
        df = pd.read_csv(filename)
        dfs.append(df)
        for metric in fairness_metrics:
            print("{}".format(round(df[metric].iloc[-1], 2)))

    df_concat = pd.concat((dfs))
    df_concat = df_concat.groupby(df_concat.index)
    df_concat_avg = df_concat.mean()
    df_concat_std = df_concat.std()
    for metric in metrics_results:
        print("{} - {}: {}+-{}".format(fl, metric, round(df_concat_avg[metric].iloc[-1], 2), round(df_concat_std[metric].iloc[-1], 2)))

    return df_concat_avg


def get_filename(dataset_name, run_num, fl, is_baseline):
    if is_baseline:
        filename = './datasets/{}/run_{}/{}.csv'.format(dataset_name, run_num, fl)
    else:
        filename = './datasets/{}/run_{}/{}.csv'.format(dataset_name, run_num, fl)

    return filename


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
