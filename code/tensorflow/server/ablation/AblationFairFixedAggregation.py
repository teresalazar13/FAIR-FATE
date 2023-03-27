from code.tensorflow.server.FairFateAggregation import FairFateAggregation


class AblationFairFixedAggregation(FairFateAggregation):

    def __init__(self, state, dataset, aggregation_metrics, l=0.05):
        self.l = l
        super().__init__(state, dataset, aggregation_metrics)

    # Overrides FAIR-FATE method
    def get_beta(self):
        return None

    # Overrides FAIR-FATE method
    def get_lambda(self):
        return self.l
