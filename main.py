from code.datasets.Adult import Adult
from code.datasets.Compas import Compas
from code.plots.plot import plot_avg_results
from code.plots.pie_chart import create_stats_sensitive_distribution_all
from code.run import run

if __name__ == '__main__':
    # dataset = Adult()
    dataset = Compas()

    create_stats_sensitive_distribution_all(
        dataset, "./datasets/{}".format(dataset.name)
    )

    run(dataset, 50, 10)
    # plot_avg_results(dataset.name, 5)

    """
    for alpha in [0.1, 0.2, 0.5, 5000]:
        run(dataset, 50, 10, alpha)"""
