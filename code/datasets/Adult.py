from code.datasets.Dataset import Dataset
from code.datasets.Feature import Feature


class Adult(Dataset):
    def __init__(self):
        name = "adult"
        sensitive_attributes = [Feature("gender", ["Male"], ["Female"], "Male", "Female")]
        # [Feature("race", ["White"], ["Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])]
        target = Feature("income", ">50K", "<=50K")
        cat_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race",
                       "native-country"]
        all_columns = ["age", "workclass", "fnlwgt", "education", "educational-num", "marital-status", "occupation",
                       "relationship", "race", "gender", "capital-gain", "capital-loss", "hours-per-week",
                       "native-country"]
        number_of_clients = 15
        num_clients_per_round = 5
        metric = "SP"
        super().__init__(name, sensitive_attributes, target, cat_columns, all_columns, number_of_clients,
                         num_clients_per_round, metric)

    def custom_preprocess(self, df):
        return df
