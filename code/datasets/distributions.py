import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_x_dirichlet(seed, alpha, dataset, x_train, y_train):
    num_clients = dataset.number_of_clients
    num_classes = len(dataset.combs)
    np.random.seed(seed)
    s = np.random.dirichlet(np.ones(num_clients) * alpha, num_classes)
    plot_distributions(num_clients, dataset.combs, s)

    df = join_x_and_y(dataset, x_train, y_train)
    df.sample(frac=1, random_state=seed)  # shuffle

    x_train_dirichlet = [[] for _ in range(num_clients)]
    y_train_dirichlet = [[] for _ in range(num_clients)]
    for comb_idx in range(len(dataset.combs)):
        df_temp = df.copy(deep=True)
        for k, [v, _, _] in dataset.combs[comb_idx].items():
            df_temp = df_temp[df_temp[k] == v]

        size = len(df_temp)
        start = 0
        for client_idx in range(num_clients):
            if client_idx != num_clients - 1:
                n_instances_for_client = round(size * s[comb_idx][client_idx])
                df_client = df_temp.iloc[start:start+n_instances_for_client]
                start = start + n_instances_for_client
            else:   # get the rest
                df_client = df_temp.iloc[start:]
            x_train_client = df_client[dataset.all_columns].to_numpy().tolist()
            y_train_client = df_client[dataset.target.name].to_numpy()
            x_train_dirichlet[client_idx].extend(x_train_client)
            y_train_dirichlet[client_idx].extend(y_train_client)

    # shuffle
    for i in range(len(x_train_dirichlet)):
        x_train_dirichlet[i] = np.array(x_train_dirichlet[i])
        y_train_dirichlet[i] = np.array(y_train_dirichlet[i])
        p = np.random.permutation(len(x_train_dirichlet[i]))
        x_train_dirichlet[i] = x_train_dirichlet[i][p]
        y_train_dirichlet[i] = y_train_dirichlet[i][p]

    return x_train_dirichlet, y_train_dirichlet


def plot_distributions(num_clients, combs, s):
    sum = 0
    for i in range(len(combs)):
        label = ""
        for k, v in combs[i].items():
            label += "{}: {};".format(k, v[-1])
        plt.barh(range(num_clients), s[i], left=sum, label=label[:-1])
        sum += s[i]
    plt.yticks([i for i in range(num_clients)], ["Client {}".format(i + 1) for i in range(num_clients)])
    plt.xlabel("Sensitive attribute distribution")
    plt.legend()
    # plt.show()


def join_x_and_y(dataset, x_train, y_train):
    df = pd.DataFrame(data=x_train, columns=dataset.all_columns)
    df[dataset.target.name] = y_train

    return df
