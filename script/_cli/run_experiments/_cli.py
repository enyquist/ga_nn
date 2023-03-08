# standard libaries
import subprocess
import logging
from tqdm import tqdm

# third party libraries
import click

logger = logging.getLogger(__name__)

@click.command()
@click.option("--runs", type=str, default="5", help="Number of runs to complete of each experiment.")
def run_experiments(runs: str) -> None:
    """Train multiple neural networks with Genetic Algorithms."""
    num_runs = int(runs)
    for run in tqdm(range(num_runs), desc="Runs"):
        logger.info("Running experiments...")

        subprocess.run(["ga", "train", "--dataset", "iris"])
        logger.info(f"Iris Run # {run + 1} Complete")

        subprocess.run(["ga", "train", "--dataset", "wine"])
        logger.info(f"Wine Run # {run + 1} Complete")

        subprocess.run(["ga", "train", "--dataset", "seeds"])
        logger.info(f"Seeds Run # {run + 1} Complete")
