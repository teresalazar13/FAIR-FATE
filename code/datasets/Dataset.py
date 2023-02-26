from abc import abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import itertools


class Dataset:
    def __init__(self, name, sensitive_attributes, target, cat_columns, all_columns, number_of_clients,
                 num_clients_per_round, num_epochs, learning_rate):
        self.name = name
        self.sensitive_attributes = sensitive_attributes
        df = pd.read_csv('./datasets/{}/{}.csv'.format(self.name, self.name))
        self.sensitive_idx = [df.columns.get_loc(s.name) for s in sensitive_attributes]
        self.target = target
        self.cat_columns = cat_columns
        self.all_columns = all_columns
        self.n_features = len(all_columns)
        self.number_of_clients = number_of_clients
        self.num_clients_per_round = num_clients_per_round
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.combs = self.create_combinations(True)
        self.combs_without_target = self.create_combinations(False)

    def preprocess(self):
        df = pd.read_csv('./datasets/{}/{}.csv'.format(self.name, self.name))
        df = self.custom_preprocess(df)

        # Convert to categorical
        for s in self.sensitive_attributes:
            positive = df[s.name].isin(s.positive)
            negative = df[s.name].isin(s.negative)
            df.loc[positive, s.name] = 1.0
            df.loc[negative, s.name] = 0.0

        positive = df[self.target.name] == self.target.positive
        negative = df[self.target.name] == self.target.negative
        df.loc[positive, self.target.name] = 1.0
        df.loc[negative, self.target.name] = 0.0

        if self.name == "law":  # already clean
            return df

        df[self.cat_columns] = df[self.cat_columns].astype('category')
        df[self.cat_columns] = df[self.cat_columns].apply(lambda x: x.cat.codes)

        # Normalize
        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        columns = df.columns
        df = pd.DataFrame(x_scaled)
        df.columns = columns

        return df

    def train_val_test_split(self, seed):
        df = self.preprocess()
        X = df.loc[:, df.columns != self.target.name].to_numpy()
        y = df[self.target.name].to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=seed)
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.int32)
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.int32).reshape(len(y_test), 1)
        x_val = x_val.astype(np.float32)
        y_val = y_val.astype(np.int32).reshape(len(y_val), 1)

        return x_train, y_train, x_test, y_test, x_val, y_val


    def _calculate_sp(self, df, dataset):
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

        print("S_0", S_0)
        print("Y_1_S_0", Y_1_S_0)
        print("S_1", S_1)
        print("Y_1_S_1", Y_1_S_1)

        if S_0 == 0 or S_1 == 0:
            return 0

        if (Y_1_S_1 / S_1) == 0:
            return 0
        return round((Y_1_S_0 / S_0) / (Y_1_S_1 / S_1), 3)

    # for global reweighting baseline
    def get_weights_global(self, x_ys, idx):
        df = pd.DataFrame(data=[], columns=self.all_columns + [self.target.name])
        dfs = []
        for x, y, title in x_ys:
            df_new = pd.DataFrame(data=np.concatenate((x, np.stack(y, axis=0)), axis=1),
                                  columns=self.all_columns + [self.target.name])
            df = pd.concat([df, df_new], ignore_index=True)
            dfs.append(df_new)

        reweighting_weights = [[] for _ in range(self.number_of_clients)]
        for i in range(len(idx)):
            reweighting_weights[idx[i]] = [0 for _ in range(len(x_ys[i][0]))]

        for comb in self.combs:
            weight = get_weight(df, comb)

            for i in range(len(idx)):
                indexes_for_weight = get_indexes_for_weight(dfs[i], comb)

                for index in indexes_for_weight:
                    reweighting_weights[idx[i]][index] = weight

        return reweighting_weights

    # for local reweighting baseline
    def get_weights_local(self, x_ys):
        dfs = []
        for x, y, title in x_ys:
            df_new = pd.DataFrame(data=np.concatenate((x, np.stack(y, axis=0)), axis=1),
                                  columns=self.all_columns + [self.target.name])
            dfs.append(df_new)

        reweighting_weights = [[0 for _ in range(len(x_ys[i][0]))] for i in range(self.number_of_clients)]

        for i in range(len(x_ys)):
            for comb in self.combs:
                weight = get_weight(dfs[i], comb)
                indexes_for_weight = get_indexes_for_weight(dfs[i], comb)

                for index in indexes_for_weight:
                    reweighting_weights[i][index] = weight

        return reweighting_weights

    # Returns something like: [{'income': 1.0, 'gender': 1.0}, {'income': 1.0, 'gender': 0.0}, {'income': 0.0,
    # 'gender': 1.0}, {'income': 0.0, 'gender': 0.0}]
    def create_combinations(self, with_target):
        if with_target:
            classes = [[
                {self.target.name: [1.0, self.target.positive, self.target.positive_label]},
                {self.target.name: [0.0, self.target.negative, self.target.negative_label]}
            ]]
        else:
            classes = []

        for s in self.sensitive_attributes:
            classes.append([
                {s.name: [1.0, s.positive, s.positive_label]},
                {s.name: [0.0, s.negative, s.negative_label]}
            ])

        combs_tuples = itertools.product(*classes)
        combs = []
        for comb_tuples in combs_tuples:
            comb = {}
            for d in comb_tuples:
                comb.update(d)
            combs.append(comb)

        return combs

    @abstractmethod
    def custom_preprocess(self, df):
        raise NotImplementedError("Must override update")


# comb = {"gender": "Female", "income": ">50K"}
def get_weight(df, comb):
    df_temp = df.copy(deep=True)
    expected = 1

    for attribute_name, [attribute_value, _, _] in comb.items():
        df_temp = df_temp[df_temp[attribute_name] == attribute_value]
        expected = expected * (len(df[df[attribute_name] == attribute_value]) / len(df))

    observed = len(df_temp) / len(df)
    if observed == 0 or expected == 0:
        weight = 1.0
    else:
        weight = expected / observed

    return weight


def get_indexes_for_weight(df_i, comb):
    df_temp = df_i.copy(deep=True)

    for k, [v, _, _] in comb.items():
        df_temp = df_temp[df_temp[k] == v]

    return df_temp.index.values.tolist()
