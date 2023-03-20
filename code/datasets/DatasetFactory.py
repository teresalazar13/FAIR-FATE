from code.datasets.Adult import Adult
from code.datasets.Compas import Compas
from code.datasets.Dutch import Dutch
from code.datasets.Law import Law


def get_datasets():
    return [
        Adult(), Compas(), Dutch(), Law()
    ]


def get_datasets_names():
    return [dataset.name for dataset in get_datasets()]


def get_dataset(dataset_name):
    for dataset in get_datasets():
        if dataset.name == dataset_name:
            return dataset
    raise ValueError(format)
