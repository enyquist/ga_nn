# standard libaries
import logging

# third party libaraies
import click

# local libraries
from script._cli.train._cli import train, train_base_model
from script._cli.run_experiments._cli import run_experiments


logger = logging.getLogger(__name__)

@click.group()
def ga() -> None:
    """Main function."""
    pass


ga.add_command(train)
ga.add_command(train_base_model)
ga.add_command(run_experiments)


def main() -> None:
    """
    Main Entry Point
    """
    ga()


if __name__ == "__main__":
    main()
