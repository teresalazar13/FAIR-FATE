from code.algorithms.FairFate import FairFate
from code.algorithms.FedAvg import FedAvg
from code.algorithms.FedAvgGR import FedAvgGR
from code.algorithms.FedAvgLR import FedAvgLR
from code.algorithms.FedDemon import FedDemon
from code.algorithms.FedMom import FedMom
from code.algorithms.FedVal import FedVal
from code.algorithms.ablation.AblationFairDemonFixed import AblationFairDemonFixed
from code.algorithms.ablation.AblationFairDemonLinear import AblationFairDemonLinear
from code.algorithms.ablation.AblationFairExponential import AblationFairExponential
from code.algorithms.ablation.AblationFairFixed import AblationFairFixed
from code.algorithms.ablation.AblationFairLinear import AblationFairLinear
from code.algorithms.ablation.AblationFairMomExponential import AblationFairMomExponential
from code.algorithms.ablation.AblationFairMomFixed import AblationFairMomFixed
from code.algorithms.ablation.AblationFairMomLinear import AblationFairMomLinear


def get_algorithms():
    return [
        FairFate(), FedAvg(), FedAvgGR(), FedAvgLR(), FedDemon(), FedMom(), FedVal(), AblationFairDemonFixed(),
        AblationFairDemonLinear(), AblationFairExponential(), AblationFairFixed(), AblationFairLinear(),
        AblationFairMomExponential(), AblationFairMomFixed(), AblationFairMomLinear()
    ]

def get_algorithms_names():
    return [algorithm.name for algorithm in get_algorithms()]


def get_algorithm(algorithm_name):
    for alg in get_algorithms():
        if alg.name == algorithm_name:
            return alg
    raise ValueError(format)
