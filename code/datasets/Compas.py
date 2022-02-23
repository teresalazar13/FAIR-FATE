from code.datasets.Dataset import Dataset
from code.datasets.Feature import Feature


# https://search.r-project.org/CRAN/refmans/fairness/html/compas.html
# https://www.kaggle.com/danofer/compass
# https://github.com/anonymous12138/biasmitigation/tree/main/Data
# https://github.com/algofairness/fairness-comparison/blob/master/fairness/data/preprocessed/propublica-recidivism_processed.csv
# https://investigate.ai/propublica-criminal-sentencing/week-5-1-machine-bias-class/
# https://aif360.readthedocs.io/en/latest/modules/generated/aif360.datasets.CompasDataset.html

class Compas(Dataset):
    def __init__(self):
        name = "compas"
        sensitive_attributes = [
            Feature(
              "race", ["Caucasian"], ["African-American", "Hispanic", "Other", "Asian", "Native American"],
              "Caucassian", "Non-Caucassian")
        ]
        target = Feature("two_year_recid", 0, 1, "won't recidivate", "will recidivate")
        # The label value 0 in this case is considered favorable (no recidivism)
        cat_columns = ["sex", "c_charge_degree", "c_charge_desc"]
        all_columns = ["sex", "age", "race", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count",
                       "c_charge_degree", "c_charge_desc", "decile_score"]
        number_of_clients = 10
        num_clients_per_round = 3
        num_epochs = 10
        learning_rate = 0.01
        super().__init__(name, sensitive_attributes, target, cat_columns, all_columns, number_of_clients,
                         num_clients_per_round, num_epochs, learning_rate)

    def custom_preprocess(self, df):
        return df.drop(["sex-race", "age_cat", "score_text"], axis=1)
