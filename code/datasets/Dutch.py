from code.datasets.Dataset import Dataset
from code.datasets.Feature import Feature


class Dutch(Dataset):
    def __init__(self):
        name = "dutch"
        sensitive_attributes = [Feature("sex", ["male"], ["female"], "Male", "Female")]
        target = Feature("occupation", 1, 0, "'high level'", "'low level'")
        cat_columns = []
        all_columns = [
            "sex", "age", "household_position", "household_size", "prev_residence_place", "citizenship",
            "country_birth", "edu_level", "economic_status","cur_eco_activity", "Marital_status"
        ]
        number_of_clients = 20
        num_clients_per_round = 6
        num_epochs = 10
        learning_rate = 0.01
        super().__init__(name, sensitive_attributes, target, cat_columns, all_columns, number_of_clients,
                         num_clients_per_round, num_epochs, learning_rate)

    def custom_preprocess(self, df):
        return df
