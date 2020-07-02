'''
Simple command-line interface to run PIE experiment.
'''

import click

import PIE


# -- Commands -----------------------------------------------------------------

@click.command(name='run')
@click.argument(
    'configfile',
    type=click.Path(file_okay=True, dir_okay=False, exists=True)
)
def run_experiment(configfile):
    """Run a full image analysis experiment."""
    PIE.run_growth_rate_analysis(configfile)


# -- Create command group -----------------------------------------------------

@click.group()
def cli():  # pragma: no cover
    """Command line interface for PIE."""
    pass


cli.add_command(run_experiment)
