# standard library
import joblib
from uuid import uuid4
from pathlib import Path
import random

# third party libraries
import click
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_genetic.space import Categorical, Continuous
from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space

# local libraries
from script.util import generate_hidden_layer_combinations, load_data, scoring_func, create_log_file

import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__name__).resolve().parents[0]


@click.command()
@click.option("--dataset", type=str, default="iris", help="Name of the dataset. Options are 'iris', 'wine', and 'seeds'.")
def train_base_model(dataset: str) -> None:
    """Train a base MLP."""

    # Set up logging
    model_uuid = uuid4()
    logger_path = REPO_ROOT / "models" / dataset / "logs"
    logger_path.mkdir(parents=True, exist_ok=True)
    logger = create_log_file(logger_path / f"{model_uuid}.log")

    # Load the data
    X, y = load_data(dataset=dataset)

    # Split the data
    random_number = random.randint(0, 100)
    logger.info(f"Run Random Number: {random_number}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_number)

    # The base classifier to tune
    clf = MLPClassifier(
        early_stopping=True,
        learning_rate="adaptive",
    )

     # Train and optimize the estimator
    clf.fit(X_train, y_train)
    y_predict_ga = clf.predict(X_test)

    logger.info(f"Test Accuracy: {accuracy_score(y_test, y_predict_ga)*100:.3f}%")
