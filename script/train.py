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
def main(dataset: str) -> None:
    """Main function."""


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

    # Define parameter grid
    param_grid = {
        "clf__estimator__hidden_layer_sizes": Categorical(generate_hidden_layer_combinations()),
        "clf__estimator__activation": Categorical(["logistic", "tanh", "relu"]),
        "clf__estimator__alpha": Continuous(1e-5, 2e-5),
        "clf__estimator__tol": Continuous(1e-2, 1e10, distribution='log-uniform'),
        "clf__estimator__early_stopping": Categorical([True]),
    }

    # The base classifier to tune
    clf = MLPClassifier(hidden_layer_sizes=(50, 30))
    pipe = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('clf', OneVsRestClassifier(estimator=clf)),
        ]
    )

    # Define the model
    evolved_estimator = GASearchCV(
        estimator=pipe,
        cv=5,
        scoring=scoring_func,
        param_grid=param_grid,
        n_jobs=-1,
        verbose=True,
        population_size=100,
        generations=200,
        error_score="raise",
    )

    callbacks = [
        ConsecutiveStopping(generations=15, metric='fitness')
    ]

     # Train and optimize the estimator
    evolved_estimator.fit(X_train, y_train, callbacks=callbacks)
    y_predict_ga = evolved_estimator.predict(X_test)

    logger.info(f"Test Accuracy: {accuracy_score(y_test, y_predict_ga)*100:.3f}%")
    logger.info(f"Best Parameters: {evolved_estimator.best_params_}")

    # save model params with joblib
    out_dir = REPO_ROOT / "models" / dataset / model_uuid
    joblib.dump(evolved_estimator.best_params_, out_dir / f"{model_uuid}.joblib")

    # Plot the fitness evolution
    plot_fitness_evolution(
        evolved_estimator,
    )

    plt.savefig(out_dir / f"{model_uuid}_fitness_evolution.png")

    # Plot the search space
    plot_search_space(
        evolved_estimator,
    )

    plt.savefig(out_dir / f"{model_uuid}_search_space.png")


if __name__ == "__main__":
    main()
