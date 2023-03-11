# standard libaries
import subprocess
import logging
from tqdm import tqdm

# third party libraries
import click

logger = logging.getLogger(__name__)

@click.command()
@click.option("--ga", type=bool, default=True, help="Use Genetic Algorithms or not.")
@click.option("--runs", type=int, default=5, help="Number of runs to complete of each experiment.")
def run_experiments(ga: bool = True, runs: int = 5) -> None:
    """Train multiple neural networks with or without Genetic Algorithms."""
    for run in tqdm(range(runs), desc="Runs"):
        logger.info("Running experiments...")

        if ga:
            subprocess.run(["ga", "train", "--dataset", "iris"])
            logger.info(f"Iris Run # {run + 1} Complete")

            subprocess.run(["ga", "train", "--dataset", "wine"])
            logger.info(f"Wine Run # {run + 1} Complete")

            subprocess.run(["ga", "train", "--dataset", "seeds"])
            logger.info(f"Seeds Run # {run + 1} Complete")

        else:
            subprocess.run(["ga", "train-base-model", "--dataset", "iris"])
            logger.info(f"Iris Run # {run + 1} Complete")

            subprocess.run(["ga", "train-base-model", "--dataset", "wine"])
            logger.info(f"Wine Run # {run + 1} Complete")

            subprocess.run(["ga", "train-base-model", "--dataset", "seeds"])
            logger.info(f"Seeds Run # {run + 1} Complete")