from code.tensorflow.server.FairFateAggregation import FairFateAggregation


class AblationFairDemonLinearAggregation(FairFateAggregation):

    def __init__(self, state, dataset, aggregation_metrics, beta0=0.9, eta=0.05, MAX=10000, l0=0.1):
        super().__init__(state, dataset, aggregation_metrics, beta0=beta0)
        self.eta = eta
        self.MAX = MAX
        self.l0 = l0

    # Overrides FAIR-FATE method
    def get_lambda(self):
        lambda_ = self.l0 + self.iteration*self.eta
        if lambda_ > self.MAX:
            lambda_ = self.MAX

        return lambda_
