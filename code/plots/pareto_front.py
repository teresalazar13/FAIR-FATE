import matplotlib.pyplot as plt
import random
from random import randint

from code.plots.plot import get_dfs


def plot_pareto_fronts(dataset_name, num_runs, num_rounds, fls_fair_fate_alpha_metric, metric_a, hyperparameters, hyperparameter_name):
    metrics_results = ["ACC", "F1Score", "MCC", "SP_ratio", "TPR_ratio", "EQO_ratio"]
    plot_index = 1
    plt.figure(figsize=(7, 7))

    labels = [r'${}={}$'.format(hyperparameter_name[0], l) for l in hyperparameters]
    colors = get_random_colors(len(hyperparameters), len(hyperparameter_name[0]))

    for fls_fair_fate in fls_fair_fate_alpha_metric:
        alpha, metrics_F, fls = fls_fair_fate
        dfs_fair_fate = get_dfs(num_rounds, fls, dataset_name, num_runs, metrics_results, metrics_F, False)
        plot_pareto_front(
            dfs_fair_fate, fls, metrics_F, metric_a, alpha, plot_index, hyperparameters, hyperparameter_name, colors
        )
        plot_index += 1

    plt.subplots_adjust(hspace=0.6, wspace=0.45)
    handles = [plt.plot([], [], color=colors[i], marker="o", ls="")[0] for i in range(len(colors))]
    plt.legend(
        handles=handles, labels=labels, loc=(-2.45, -0.65), prop={'size': 12}, ncol=len(handles)
    )
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    filename = "pareto_{}-{}".format(hyperparameter_name[1], dataset_name)
    plt.savefig('./datasets/{}/{}.png'.format(dataset_name, filename), bbox_inches='tight', dpi=300)
    #plt.show()


def get_random_colors(size, seed):
    colors = []
    random.seed(seed)

    for i in range(size):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    return colors


def plot_pareto_front(dfs, fls, metric_F, metric_a, alpha, plot_index, hyperparameters, hyperparameter_name, colors):
    x = []
    y = []
    costs = []
    labels = []
    values = []

    for i in range(len(dfs)):
        value = fls[i].split(hyperparameter_name[1])[1].split("_")[0][1:]  # get hyperparameter value from file name
        values.append(float(value))
        labels.append(r'$\{}={}$'.format(hyperparameter_name[0], value))

        value_a = dfs[i][metric_a].iloc[-1]
        value_b = dfs[i][metric_F].iloc[-1]
        x.append(value_a)
        y.append(value_b)
        costs.append([x, y])

    metric_fairness = metric_F[0].replace("_ratio", "").replace("TPR", "EO")
    if alpha:
        title = r'$\sigma={}$, $F$={}'.format(alpha, metric_fairness)
    else:
        title = r'RND, $F$={}'.format(metric_fairness)
    plt.subplot(3, 3, plot_index)
    plt.title(title, fontsize=12)
    plt.xlabel(metric_a.replace("_ratio", ""), fontsize=12)
    plt.ylabel(metric_fairness, fontsize=12)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    for i in range(len(x)):
        value = values[i]
        rho_index = hyperparameters.index(value)
        plt.scatter(x[i], y[i], color=colors[rho_index])
