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
    PIE.run_growth_rate_analysis(analysis_config_file = configfile)


@click.command(name='track_single_position')
@click.argument(
    'xy_pos_idx',
    type=int
)
@click.argument(
    'configfile',
    type=click.Path(file_okay=True, dir_okay=False, exists=True)
)
def track_single_pos(xy_pos_idx, configfile):
    """Analyze a single image using PIE.

    XY_POS_IDX is an integer corresponding to xy position to be analyzed
    
    CONFIGFILE is the path to configuration setup file for this
    experiment
    """
    PIE.track_colonies_single_pos(
        xy_pos_idx,
        analysis_config_file = configfile)

# -- Create command group -----------------------------------------------------

@click.group()
def cli():  # pragma: no cover
    """Command line interface for PIE."""
    pass


cli.add_command(run_experiment)
cli.add_command(track_single_pos)
