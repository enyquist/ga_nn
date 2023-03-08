# standard library
from random import randint
import itertools

# third party libraries
import numpy as np

# local libraries
from script.util import generate_hidden_layer_combinations, max_loss


def test_returns_list_of_tuples():
    """
    Test that the function returns a list of tuples.
    """

    result = generate_hidden_layer_combinations()
    assert isinstance(result, list)
    for combination in result:
        assert isinstance(combination, tuple)
        for neuron_count in combination:
            assert isinstance(neuron_count, int)


def test_combinations_contain_integers_within_range():
    """
    Test whether the combinations contain integers within the specified range.
    """

    min_neurons_per_layer = 16
    max_neurons_per_layer = 256
    result = generate_hidden_layer_combinations()
    for combination in result:
        for neuron_count in combination:
            assert min_neurons_per_layer <= neuron_count <= max_neurons_per_layer

def test_max_hidden_layers():
    """
    Test whether the max number of hidden layers is respected.
    """
    
    max_hidden_layers = 3
    result = generate_hidden_layer_combinations(max_hidden_layers=max_hidden_layers)
    possible_combinations = sum([len(list(itertools.product(range(16, 256 + 1, 16), repeat=i))) for i in range(1, max_hidden_layers + 1)])
    assert len(result) == possible_combinations


def test_max_log_loss():
    """
    Test whether the max log loss is calculated for given targets y
    """
    
    y = [0, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    assert round(max_loss(y), 0) == 36.0
