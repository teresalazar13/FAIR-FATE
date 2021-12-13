import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_stats_sensitive_distribution_all(dataset, plot_filename):
    df = dataset.preprocess()
    labels = []
    sizes = []

    for comb in dataset.combs:
        df_temp = df.copy(deep=True)
        label = ""

        for k, [v, _, t_] in comb.items():
            df_temp = df_temp[df_temp[k] == v]
            label += "{}: {}\n".format(k, t_)

        sizes.append(len(df_temp))
        labels.append(label)

    sp = calculate_sp(df, dataset)
    plot_pie_chart("", sp, labels, sizes, plot_filename)


def create_stats_sensitive_distribution(x_ys, dataset, plot_filename):
    for x, y, title in x_ys:
        df = pd.DataFrame(data=np.concatenate((x, np.stack(y, axis=0)), axis=1),
                          columns=dataset.all_columns + [dataset.target.name])
        labels = []
        sizes = []

        for comb in dataset.combs:
            df_temp = df.copy(deep=True)
            label = ""

            for k, [v, _, t_] in comb.items():
                df_temp = df_temp[df_temp[k] == v]
                label += "{}: {}\n".format(k, t_)

            sizes.append(len(df_temp))
            labels.append(label)

        sp = calculate_sp(df, dataset)
        plot_pie_chart(title, sp, labels, sizes, plot_filename)


def calculate_sp(df, dataset):
    df_temp_unpriv = df.copy(deep=True)
    df_temp_priv = df.copy(deep=True)

    for s in dataset.sensitive_attributes:
        df_temp_unpriv = df_temp_unpriv[df_temp_unpriv[s.name] == 0.0]
        df_temp_priv = df_temp_priv[df_temp_priv[s.name] == 1.0]

    S_0 = len(df_temp_unpriv)
    S_1 = len(df_temp_priv)
    df_temp_unpriv = df_temp_unpriv[df_temp_unpriv[dataset.target.name] == 1.0]
    df_temp_priv = df_temp_priv[df_temp_priv[dataset.target.name] == 1.0]
    Y_1_S_0 = len(df_temp_unpriv)
    Y_1_S_1 = len(df_temp_priv)

    if S_0 == 0 or S_1 == 0:
        return 0

    return round((Y_1_S_0 / S_0) / (Y_1_S_1 / S_1), 3)


# Plot pie chart
def plot_pie_chart(title, sp, labels, sizes, plot_filename):
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', labeldistance=1.2, shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title("SP: {}".format(sp))

    plt.savefig('{}/plot_{}.png'.format(plot_filename, title))
    #plt.show()
