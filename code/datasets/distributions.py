import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_x_dirichlet(random_state, alpha, dataset, x_train, y_train):
    num_clients = dataset.number_of_clients
    num_classes = len(dataset.combs_without_target)
    np.random.seed(random_state)
    s = np.random.dirichlet(np.ones(num_clients) * alpha, num_classes)
    plot_distributions(num_clients, num_classes, s)

    df = join_x_and_y(dataset, x_train, y_train)
    df.sample(frac=1)  # shuffle

    x_train_dirichlet = [[] for _ in range(num_clients)]
    y_train_dirichlet = [[] for _ in range(num_clients)]
    for comb_idx in range(len(dataset.combs_without_target)):
        df_temp = df.copy(deep=True)
        for k, [v, _, _] in dataset.combs_without_target[comb_idx].items():
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

    return np.array(sum(x_train_dirichlet, [])), np.array(sum(y_train_dirichlet, [])).reshape(1, -1).T
    # desired shape for func get_tf_train_dataset()


def plot_distributions(num_clients, num_classes, s):
    sum = 0
    for i in range(num_classes):
        plt.barh(range(num_clients), s[i], left=sum)
        sum += s[i]
    # plt.show()


def join_x_and_y(dataset, x_train, y_train):
    df = pd.DataFrame(data=x_train, columns=dataset.all_columns)
    df[dataset.target.name] = y_train

    return df
