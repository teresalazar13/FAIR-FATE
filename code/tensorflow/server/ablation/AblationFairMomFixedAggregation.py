from code.tensorflow.server.FairFateAggregation import FairFateAggregation


class AblationFairMomFixedAggregation(FairFateAggregation):

    def __init__(self, state, dataset, aggregation_metrics, beta=0.9, l=0.05):
        self.beta = beta
        self.l = l
        super().__init__(state, dataset, aggregation_metrics)

    # Overrides FAIR-FATE method
    def get_beta(self):
        return self.beta

    # Overrides FAIR-FATE method
    def get_lambda(self):
        return self.l
