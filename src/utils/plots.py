from matplotlib import pyplot as plt


def plot_epochs_history(metric_values_list: list[list], metric: str = 'Accuracy') -> None:
    """plot epochs history.

    Args:
        metric_values_list (list[list]): List (train, validation for example) where each item is a list 
        containing the metric values of each epoch.

        metric (str, optional): Metric name (usually loss and accuracy). Defaults to 'Accuracy'.
    """
    for metric_values in metric_values_list:
        plt.plot(metric_values)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.show()
