from code.tensorflow.server.FairFateAggregation import FairFateAggregation


class AblationFairDemonFixedAggregation(FairFateAggregation):

    def __init__(self, state, dataset, aggregation_metrics, beta0=0.9, l=0.5):
        super().__init__(state, dataset, aggregation_metrics, beta0=beta0)
        self.l = l

    # Overrides FAIR-FATE method
    def get_lambda(self):
        return self.l
