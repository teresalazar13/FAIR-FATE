class Metric:
    def __init__(self, name):
        self.name = name
        self.ratios = []
        self.differences = []
        self.values = []

    def reset(self):
        self.ratios = []
        self.differences = []
        self.values = []
