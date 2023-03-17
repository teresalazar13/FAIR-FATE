from code.tensorflow.server.FairFateAggregation import FairFateAggregation


class AblationFairExponentialAggregation(FairFateAggregation):

    def __init__(self, state, dataset, aggregation_metrics, MAX=1000, l0=0.1, rho=0.05):
        super().__init__(state, dataset, aggregation_metrics, MAX=MAX, l0=l0, rho=rho)

    # Overrides FAIR-FATE method
    def get_beta(self):
        return None
