import matplotlib.pyplot as plt
import random
from random import randint

from code.plots.plot import get_dfs


def plot_pareto_fronts(dataset_name, num_runs, num_rounds, fls_fair_fate_alpha_metric, metric_a, rhos_, betas, filename):
    metrics_results = ["ACC", "F1Score", "MCC", "SP_ratio", "TPR_ratio", "EQO_ratio"]
    plot_index = 1
    plt.figure(figsize=(8, 8))

    rhos_labels = [r'$\rho={}$'.format(l) for l in rhos_]
    betas_labels = [r'$\beta={}$'.format(b) for b in betas]
    colors = get_random_colors(len(rhos_))
    markers = ["o", "+", "*", "x", "v"]

    for fls_fair_fate in fls_fair_fate_alpha_metric:
        alpha, metrics_F, fls = fls_fair_fate
        dfs_fair_fate = get_dfs(num_rounds, fls, dataset_name, num_runs, metrics_results, metrics_F, False)
        plot_pareto_front(dfs_fair_fate, fls, metrics_F, metric_a, alpha, plot_index, rhos_, betas, colors, markers)
        plot_index += 1

    plt.tight_layout(h_pad=0.75, w_pad=0.75)

    if dataset_name == "compas":
        coords = [(-2.63, -0.63), (-1.6, -0.45)]
    else:
        coords = [(-2.45, -0.63), (-1.48, -0.45)]
    rho_handles = [plt.plot([], [], color=colors[i], marker="o", ls="")[0] for i in range(len(colors))]
    rho_legend = plt.legend(handles=rho_handles, labels=rhos_labels, loc=coords[0], prop={'size': 11}, ncol=len(rho_handles))
    plt.gca().add_artist(rho_legend)
    beta_handles = [plt.plot([], [], color="black", marker=markers[i], ls="")[0] for i in range(len(markers[:len(betas_labels)]))]
    beta_legend = plt.legend(handles=beta_handles, labels=betas_labels, loc=coords[1], prop={'size': 11}, ncol=len(betas_labels))
    plt.gca().add_artist(beta_legend)

    plt.savefig('./datasets/{}/{}.png'.format(dataset_name, filename), bbox_inches='tight')
    # plt.show()


def get_random_colors(size):
    colors = []
    random.seed(10)

    for i in range(size):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    return colors


def plot_pareto_front(dfs, fls, metric_F, metric_a, alpha, plot_index, rhos_set, betas_set, colors, markers):
    x = []
    y = []
    costs = []
    labels = []
    rhos_ = []
    betas = []

    for i in range(len(dfs)):
        label_split = fls[i].split("_")
        rho_ = label_split[3][1:]
        beta = label_split[5]
        rhos_.append(float(rho_))
        betas.append(float(beta))
        labels.append(r'$\rho={}-\beta={}$'.format(rho_, beta))

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
        rho_ = rhos_[i]
        rho_index = rhos_set.index(rho_)
        beta = betas[i]
        beta_index = betas_set.index(beta)
        plt.scatter(x[i], y[i], color=colors[rho_index], marker=markers[beta_index])
