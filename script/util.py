# standard library
import itertools
import logging
from pathlib import Path
from typing import List, Tuple

# third party libraries
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.datasets import load_iris, load_wine


def generate_hidden_layer_combinations(
    max_hidden_layers: int = 5,
    min_neurons_per_layer: int = 16, 
    max_neurons_per_layer: int = 256,
) -> List[Tuple[int]]:
    """
    Generate all possible combinations of hidden layer sizes.

    Args:
        max_hidden_layers (int, optional): Max number of hidden layers. Defaults to 5.
        min_neurons_per_layer (int, optional): Minimum number of neurons in each hidden layer. Defaults to 16.
        max_neurons_per_layer (int, optional): Maximum number of neurons in each hidden layer. Defaults to 256.

    Returns:
        List[Tuple[Integer]]: List of all possible combinations of hidden layer sizes.
    """
    # Define the range of hidden layer sizes
    num_possible_hidden_layers = range(1, max_hidden_layers + 1)
    hidden_layer_sizes = range(min_neurons_per_layer, max_neurons_per_layer + 1, 16)

    # Generate all possible combinations of hidden layer sizes
    hidden_layer_combinations = []
    for length in num_possible_hidden_layers:
        for combination in itertools.product(hidden_layer_sizes, repeat=length):
            hidden_layer_combinations.append(combination)

    # Return the combinations
    return hidden_layer_combinations


def load_data(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the data.

    Args:
        dataset (str): Name of the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The data and the labels.
    """
    # Load the data
    if dataset == "iris":
        X, y = load_iris(return_X_y=True)
    elif dataset == "wine":
        X, y = load_wine(return_X_y=True)
    elif dataset == "seeds":
        X, y = load_seeds(return_X_y=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Options are 'iris', 'wine', and 'seeds'.")

    # Return the data
    return X, y


def load_seeds(return_X_y: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the seeds dataset.

    Args:
        return_X_y (bool, optional): Whether to return the data and the labels. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The data and the labels.
    """
    # Load the data
    data = np.loadtxt("data/seeds_dataset.txt", delimiter="\t")

    # Extract the features and the labels
    X = data[:, :-1]
    y = data[:, -1]

    # Return the data
    if return_X_y:
        return X, y
    else:
        return data


def scoring_func(estimator: Pipeline, X: np.ndarray, y: np.ndarray, alpha: float = 0.5) -> float:
    """
    Scoring function.

    Args:
        estimator (sklearn Pipeline): The Pipeline.
        X (np.ndarray): The data.
        y (np.ndarray): The labels.
        alpha (float, optional): The alpha parameter. Balance between favoring score and size of network. Defaults to 0.5.

    Returns:
        float: The score.
    """

    # Define variables
    max_neurons_per_layer = 256
    max_hidden_layers = 5
    max_neurons = max_hidden_layers * max_neurons_per_layer
    num_sufficient_neurons = 100  # Assume 100 neurons is sufficient for this network

    # Calculate the CE loss
    y_pred = estimator.predict_proba(X)
    loss = log_loss(y, y_pred)

    hidden_layer_sizes = estimator.named_steps["clf"].estimator.hidden_layer_sizes

    # Calculate the number of hidden layers
    hidden_layers = len(hidden_layer_sizes)

    # Calculate the number of neurons in each layer
    neurons = sum(hidden_layer_sizes)

    # Calculate the score
    score_metric = alpha * (loss + 1e-7) / max_loss()
    num_neurons_metric = (1 - alpha) * (sum([hidden_layers/max_hidden_layers, neurons/max_neurons]) / num_sufficient_neurons)
    score = sum([score_metric, num_neurons_metric])

    # Return fitness
    return 1 / score


def max_loss() -> float:
    """
    Calculate the maximum log loss for a given number of classes y.

    Args:
        y (np.ndarray): The labels.

    Returns:
        float: The maximum log loss.
    """
    # Calculate the maximum log loss
    y = [0, 1]
    y_pred = []
    for bit in y:
        y_pred.append([0, 1] if bit == 0 else [1, 0])
    max_loss = log_loss(y, y_pred)

    # Return the maximum log loss
    return max_loss



def create_log_file(log_file_name: Path) -> logging.Logger:
    """
    Create a log file.

    Args:
        log_file_name (Path): name of file to log to.

    Returns:
        logging.Logger: logger object.
    """

    # create logger
    logger = logging.getLogger(str(log_file_name))
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
