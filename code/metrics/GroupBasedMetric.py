from code.metrics.Metric import Metric


class GroupBasedMetric(Metric):
    def __init__(self, name, group_numerator, group_denominator):
        self.name = name
        super().__init__(name)

        self.group_numerator = group_numerator
        self.group_denominator = group_denominator

    def calculate(self, sensitive_attributes, df, debug=True):
        sum_ratio = 0
        sum_difference = 0

        for sensitive_attribute in sensitive_attributes:
            numerators = self.group_numerator.get_group_counts(sensitive_attribute, df)
            denominators = self.group_denominator.get_group_counts(sensitive_attribute, df)

            if 0 in numerators or 0 in denominators:
                sum_ratio += 0
                sum_difference += 1
            else:
                privileged = numerators[0] / (numerators[0] + denominators[0])
                unprivileged = numerators[1] / (numerators[1] + denominators[1])
                #print("{} - {}-{}".format(self.name, round(privileged, 2), round(unprivileged, 2)))
                sum_ratio += min(unprivileged / privileged, privileged / unprivileged)
                sum_difference += abs(privileged - unprivileged)

        ratio = round(sum_ratio / len(sensitive_attributes), 2)
        difference = round(sum_difference / len(sensitive_attributes), 2)
        self.ratios.append(ratio)
        self.differences.append(difference)

        if debug:
            print("{}_ratio - {}".format(self.name, ratio))
            print("{}_diff - {}".format(self.name, difference))

        return ratio, difference


class TP:
    @staticmethod
    def get_group_counts(sensitive_attribute, df):
        return [
            len(df[(df[sensitive_attribute] == 1) & (df["y"] == 1) & (df["y_pred"] == 1)]),
            len(df[(df[sensitive_attribute] == 0) & (df["y"] == 1) & (df["y_pred"] == 1)])
        ]


class FP:
    @staticmethod
    def get_group_counts(sensitive_attribute, df):
        return [
            len(df[(df[sensitive_attribute] == 1) & (df["y"] == 0) & (df["y_pred"] == 1)]),
            len(df[(df[sensitive_attribute] == 0) & (df["y"] == 0) & (df["y_pred"] == 1)])
        ]


class TN:
    @staticmethod
    def get_group_counts(sensitive_attribute, df):
        return [
            len(df[(df[sensitive_attribute] == 1) & (df["y"] == 0) & (df["y_pred"] == 0)]),
            len(df[(df[sensitive_attribute] == 0) & (df["y"] == 0) & (df["y_pred"] == 0)])
        ]


class FN:
    @staticmethod
    def get_group_counts(sensitive_attribute, df):
        return [
            len(df[(df[sensitive_attribute] == 1) & (df["y"] == 1) & (df["y_pred"] == 0)]),
            len(df[(df[sensitive_attribute] == 0) & (df["y"] == 1) & (df["y_pred"] == 0)])
        ]


class Sens:
    @staticmethod
    def get_group_counts(sensitive_attribute, df):
        return [
            len(df[df[sensitive_attribute] == 1]),
            len(df[df[sensitive_attribute] == 0])
        ]


class PosSens:
    @staticmethod
    def get_group_counts(sensitive_attribute, df):
        return [
            len(df[(df[sensitive_attribute] == 1) & (df["y_pred"] == 1)]),
            len(df[(df[sensitive_attribute] == 0) & (df["y_pred"] == 1)])
        ]
