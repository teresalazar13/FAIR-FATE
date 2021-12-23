from code.datasets.Adult import Adult
from code.datasets.Compas import Compas
from code.plots.plot import plot_avg_results
from code.plots.pie_chart import create_stats_sensitive_distribution_all
from code.metrics.GroupBasedMetric import GroupBasedMetric, PosSens, Sens, TP, FN, FP, TN
from code.metrics.SuperGroupBasedMetric import SuperGroupBasedMetric
from code.run import run

if __name__ == '__main__':
    #dataset = Adult()
    dataset = Compas()

    """
    create_stats_sensitive_distribution_all(
        dataset, "./datasets/{}".format(dataset.name)
    )"""

    # FAIR-FATE
    aggregation_metrics = [
        #GroupBasedMetric("SP", PosSens(), Sens()),
        GroupBasedMetric("TPR", TP(), FN()),
        #GroupBasedMetric("FPR", FP(), TN())
        #SuperGroupBasedMetric("EQO", [GroupBasedMetric("TPR", TP(), FN()), GroupBasedMetric("FPR", FP(), TN())])
    ]

    run(dataset, 50, 10, aggregation_metrics, 5000)
    #plot_avg_results(dataset.name, 10)

    """
    for alpha in [0.5, 1, 5000]:
        run(dataset, 50, 10, aggregation_metrics, alpha)"""
