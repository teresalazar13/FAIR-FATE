import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_x_dirichlet(random_state, alpha, dataset, x_train, y_train):
    num_clients = dataset.number_of_clients
    num_classes = len(dataset.sensitive_attributes) * 2
    np.random.seed(random_state)
    s = np.random.dirichlet(np.ones(num_clients) * alpha, num_classes)
    plot_distributions(num_clients, num_classes, s)

    df = join_x_and_y(dataset, x_train, y_train)
    df.sample(frac=1)  # shuffle

    x_train_dirichlet = []
    y_train_dirichlet = []
    start = 0
    for comb_idx in range(len(dataset.combs)):
        df_temp = df.copy(deep=True)
        for k, [v, _, _] in dataset.combs[comb_idx].items():
            df_temp = df_temp[df_temp[k] == v]
        size = len(df_temp)
        for client_idx in range(num_clients):
            n_instances_for_client = size * s[comb_idx][client_idx]
            df_client = df.index[start:start+n_instances_for_client]
            x_train_client = df_client[dataset.dataset.all_columns].to_numpy()
            y_train_client = df_client[dataset.dataset.target].to_numpy()
            x_train_dirichlet[client_idx].extend(x_train_client)
            y_train_dirichlet[client_idx].extend(y_train_client)
            start = start + n_instances_for_client

    return x_train_dirichlet, y_train_dirichlet


def plot_distributions(num_clients, num_classes, s):
    sum = 0
    for i in range(num_classes):
        print(np.sum(s[i]))
        plt.barh(range(num_clients), s[i], left=sum)
        sum += s[i]
    plt.show()


def join_x_and_y(dataset, x_train, y_train):
    return pd.DataFrame(data=np.concatenate((x_train, np.stack(y_train, axis=0)),
                                            columns=dataset.all_columns + [dataset.target.name]))
