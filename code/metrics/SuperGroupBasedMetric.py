from code.metrics.Metric import Metric


class SuperGroupBasedMetric(Metric):
    def __init__(self, name, group_based_metrics):
        self.name = name
        super().__init__(name)

        self.group_based_metrics = group_based_metrics

    def calculate(self, sensitive_attributes, df, debug=True):
        ratio = []
        difference = []

        for metric in self.group_based_metrics:
            metric.calculate(sensitive_attributes, df, debug=False)
            ratio.append(metric.ratios[-1])
            difference.append(metric.differences[-1])

        avg_ratio = round(avg(ratio), 2)
        avg_difference = round(avg(difference), 2)
        self.ratios.append(avg_ratio)
        self.differences.append(avg_difference)

        if debug:
            print("{}_ratio - {}".format(self.name, avg_ratio))
            print("{}_diff - {}".format(self.name, avg_difference))

        return avg_ratio, avg_difference


def avg(list_):
    if sum(list_) == 0:
        return 0
    else:
        return sum(list_) / len(list_)
