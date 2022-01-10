import matplotlib.pyplot as plt
import random
from random import randint

from code.plots.plot import get_dfs


def get_random_colors(size):
    colors = []
    random.seed(10)

    for i in range(size):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    return colors


def plot_pareto_fronts(dataset_name, num_runs, fls_fair_fate_alpha_metric, lambdas_, betas):
    metrics_results = ["ACC", "F1Score", "MCC", "SP_ratio", "TPR_ratio", "EQO_ratio"]
    plot_index = 1
    plt.figure(figsize=(8, 8))

    lambdas_labels = [r'$\lambda={}$'.format(l) for l in lambdas_]
    betas_labels = [r'$\beta={}$'.format(b) for b in betas]
    colors = get_random_colors(len(lambdas_))
    markers = ["o", "+", "*", "x", "v"]

    for fls_fair_fate in fls_fair_fate_alpha_metric:
        alpha, metric, fls = fls_fair_fate
        dfs_fair_fate = get_dfs(fls, dataset_name, num_runs, metrics_results, [metric], False)
        plot_pareto_front(dfs_fair_fate, fls, "ACC", metric, alpha, plot_index, lambdas_, betas, colors, markers)
        plot_index += 1

    plt.tight_layout(h_pad=0.75, w_pad=0.75)

    lambda_handles = [plt.plot([], [], color=colors[i], marker="o", ls="")[0] for i in range(len(colors))]
    lambda_legend = plt.legend(handles=lambda_handles, labels=lambdas_labels, loc=(-2.63, -0.63), prop={'size': 11}, ncol=len(lambda_handles))
    plt.gca().add_artist(lambda_legend)
    beta_handles = [plt.plot([], [], color="black", marker=markers[i], ls="")[0] for i in range(len(markers[:len(betas_labels)]))]
    beta_legend = plt.legend(handles=beta_handles, labels=betas_labels, loc=(-1.6, -0.45), prop={'size': 11}, ncol=len(betas_labels))
    plt.gca().add_artist(beta_legend)

    filename = './datasets/{}/pareto_fronts.png'.format(dataset_name)
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()


def plot_pareto_front(dfs, fls, metric_a, metric_b, alpha, plot_index, lambdas_set, betas_set, colors, markers):
    x = []
    y = []
    costs = []
    labels = []
    lambdas_ = []
    betas = []

    for i in range(len(dfs)):
        label_split = fls[i].split("_")
        lambda_ = label_split[3][1:]
        beta = label_split[5]
        lambdas_.append(float(lambda_))
        betas.append(float(beta))
        labels.append(r'$\lambda={}-\beta={}$'.format(lambda_, beta))

        value_a = dfs[i][metric_a].iloc[-1]
        value_b = dfs[i][metric_b].iloc[-1]
        x.append(value_a)
        y.append(value_b)
        costs.append([x, y])

    fairness_metric = metric_b.split("_")[0]
    if alpha:
        title = r'$\alpha={}$'.format(alpha)
    else:
        title = "RND"
    plt.subplot(3, 3, plot_index)
    plt.title(title)
    plt.xlabel(metric_a)
    plt.ylabel(fairness_metric)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    for i in range(len(x)):
        lambda_ = lambdas_[i]
        lambda_index = lambdas_set.index(lambda_)
        beta = betas[i]
        beta_index = betas_set.index(beta)
        plt.scatter(x[i], y[i], color=colors[lambda_index], marker=markers[beta_index])
