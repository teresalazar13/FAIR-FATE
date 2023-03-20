from code.metrics.retriever import get_aggregation_metrics


class Hyperparameters:

    def __init__(self, args):
        self.beta = self.convert_to_float_if_exists(args.beta)
        self.beta0 = self.convert_to_float_if_exists(args.beta0)
        self.rho = self.convert_to_float_if_exists(args.rho)
        self.eta = self.convert_to_float_if_exists(args.eta)
        self.l0 = self.convert_to_float_if_exists(args.l0)
        self.l = self.convert_to_float_if_exists(args.l)
        self.max = self.convert_to_float_if_exists(args.MAX)
        self.aggregation_metrics = get_aggregation_metrics(args.metrics)

    @staticmethod
    def convert_to_float_if_exists(hyperparameter):
        if hyperparameter:
            return float(hyperparameter)
        return None
