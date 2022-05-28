import matplotlib.pyplot as plt
import random
from random import randint

from code.plots.plot import get_dfs


def plot_pareto_fronts(dataset_name, num_runs, num_rounds, fls_fair_fate_alpha_metric, metric_a, hyperparameters, hyperparameter_name, filename):
    metrics_results = ["ACC", "F1Score", "MCC", "SP_ratio", "TPR_ratio", "EQO_ratio"]
    plot_index = 1
    plt.figure(figsize=(8, 8))

    labels = [r'${}={}$'.format(hyperparameter_name[0], l) for l in hyperparameters]
    colors = get_random_colors(len(hyperparameters), len(hyperparameter_name[0]))

    for fls_fair_fate in fls_fair_fate_alpha_metric:
        alpha, metrics_F, fls = fls_fair_fate
        dfs_fair_fate = get_dfs(num_rounds, fls, dataset_name, num_runs, metrics_results, metrics_F, False)
        plot_pareto_front(
            dfs_fair_fate, fls, metrics_F, metric_a, alpha, plot_index, hyperparameters, hyperparameter_name, colors
        )
        plot_index += 1

    plt.tight_layout(h_pad=1.0, w_pad=0.75)
    plt.subplots_adjust(bottom=0.102)
    handles = [plt.plot([], [], color=colors[i], marker="o", ls="")[0] for i in range(len(colors))]
    legend = plt.legend(handles=handles, labels=labels, loc=(-1.5, -0.45), prop={'size': 11}, ncol=len(handles))
    plt.gca().add_artist(legend)

    plt.savefig('./datasets/{}/{}.png'.format(dataset_name, filename), bbox_inches='tight')
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
        title = r'$\alpha$={}, $F$={}'.format(alpha, metric_fairness)
    else:
        title = r'RND, $F$={}'.format(metric_fairness)
    plt.subplot(3, 3, plot_index)
    plt.title(title)
    plt.xlabel(metric_a.replace("_ratio", ""))
    plt.ylabel(metric_fairness)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    for i in range(len(x)):
        value = values[i]
        rho_index = hyperparameters.index(value)
        plt.scatter(x[i], y[i], color=colors[rho_index])
