from code.datasets.Adult import Adult
from code.datasets.Compas import Compas
from code.plots.plot import plot_avg_results
from code.plots.pie_chart import create_stats_sensitive_distribution_all
from code.run import run

if __name__ == '__main__':
    dataset = Adult()
    # dataset = Compas()

    """
    create_stats_sensitive_distribution_all(
        dataset, "/content/gdrive/MyDrive/Colab Notebooks/{}".format(dataset.name)
    )"""

    run(dataset, 2, 3)
    # plot_avg_results(dataset.name, 1, 50)
