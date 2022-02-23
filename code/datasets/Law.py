from code.datasets.Dataset import Dataset
from code.datasets.Feature import Feature

class Law(Dataset):
    def __init__(self):
        name = "law"
        sensitive_attributes = [Feature("race", ["White"], ["Non-White"], "White", "Non-White")]
        target = Feature("pass_bar", 1.0, 0.0, "pass", "non-pass")
        cat_columns = []
        all_columns = ["decile1b", "decile3", "lsat", "ugpa", "zfygpa", "zgpa", "fulltime", "fam_inc", "male", "tier", "race"]
        number_of_clients = 12
        num_clients_per_round = 4
        num_epochs = 10
        learning_rate = 0.01
        super().__init__(name, sensitive_attributes, target, cat_columns, all_columns, number_of_clients,
                         num_clients_per_round, num_epochs, learning_rate)

    def custom_preprocess(self, df):
        return df
