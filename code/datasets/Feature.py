class Feature:
    def __init__(self, name, positive, negative, positive_label="", negative_label=""):
        self.name = name
        self.positive = positive
        self.negative = negative

        if positive_label == "":
            self.positive_label = positive
            self.negative_label = negative
        else:
            self.positive_label = positive_label
            self.negative_label = negative_label
