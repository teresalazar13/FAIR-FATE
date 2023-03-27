from code.tensorflow.server.FairFateAggregation import FairFateAggregation


class AblationFairMomLinearAggregation(FairFateAggregation):

    def __init__(self, state, dataset, aggregation_metrics, beta=0.9, MAX=1000, l0=0.1, eta=0.05):
        self.beta = beta
        self.eta = eta
        self.MAX = MAX
        self.l0 = l0
        super().__init__(state, dataset, aggregation_metrics)

    # Overrides FAIR-FATE method
    def get_beta(self):
        return self.beta

    # Overrides FAIR-FATE method
    def get_lambda(self):
        lambda_ = self.l0 + self.iteration*self.eta
        if lambda_ > self.MAX:
            lambda_ = self.MAX

        return lambda_
