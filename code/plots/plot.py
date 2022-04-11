import pandas as pd
import numpy as np


def print_avg_results(dataset_name, num_runs, fls, fls_fair_fate, fls_fedmom, fairness_metrics, metrics_results):
    fedavg_acc = get_avg_df(fls[0], dataset_name, num_runs, metrics_results, fairness_metrics, True)["ACC"].iloc[-1]
    dfs = get_dfs(fls, dataset_name, num_runs, metrics_results, fairness_metrics, True)

    # FedMom
    dfs_fedmom = get_dfs(fls_fedmom, dataset_name, num_runs, metrics_results, fairness_metrics, True)
    fairness_metrics_all = ["SP_ratio", "TPR_ratio", "EQO_ratio"]
    print_results(dfs, fls_fedmom, dfs_fedmom, fairness_metrics_all, fedavg_acc)

    # FAIR-FATE
    dfs_fair_fate = get_dfs(fls_fair_fate, dataset_name, num_runs, metrics_results, fairness_metrics, False)
    print_results(dfs, fls_fair_fate, dfs_fair_fate, fairness_metrics, fedavg_acc)


def get_dfs(fls, dataset_name, num_runs, metrics_results, fairness_metrics, is_baseline):
    dfs = []
    for fl in fls:
        df = get_avg_df(fl, dataset_name, num_runs, metrics_results, fairness_metrics, is_baseline)
        dfs.append(df)

    return dfs


def print_results(dfs, fls_alg, dfs_alg, fairness_metrics_all, fedavg_acc):
    best = [0 for _ in fairness_metrics_all]

    for df in dfs:
        for i in range(len(fairness_metrics_all)):
            value = df[fairness_metrics_all[i]].iloc[-1]
            if value > best[i]:
                best[i] = value
    best_df, best_fl = print_results_improvements(fls_alg, dfs_alg, fairness_metrics_all, best, fedavg_acc)
    print("best - {}".format(best_fl))


def print_results_improvements(fls, dfs, metrics, best, fedavg_acc):
    improvs_fair_fate = []

    print("\nImprovements:")
    for i in range(len(dfs)):
        df = dfs[i]
        improv = 0
        acc = df["ACC"].iloc[-1]

        for j in range(len(metrics)):
            value = df[metrics[j]].iloc[-1]
            improv += value - best[j]
        improv = improv / len(metrics)
        improvs_fair_fate.append(improv)
        print("F:{} | ACC:{} | {}".format(round(improv, 2), round(acc-fedavg_acc, 2), fls[i]))
    max_idx = np.argmax(improvs_fair_fate)

    return dfs[max_idx], fls[max_idx]


def get_avg_df(fl, dataset_name, num_runs, metrics_results, fairness_metrics, is_baseline):
    dfs = []
    print(fl)
    print(fairness_metrics)
    metrics_values = [0 for _ in range(num_runs)]
    for run_num in range(1, num_runs + 1):
        filename = get_filename(dataset_name, run_num, fl, is_baseline)
        df = pd.read_csv(filename)
        dfs.append(df)
        for metric in fairness_metrics:
            metrics_values[run_num - 1] += df[metric].iloc[-1]

    for run_num in range(num_runs):
        print("{}".format(round(metrics_values[run_num] / len(fairness_metrics), 2)))

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
