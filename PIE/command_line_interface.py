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


@click.command(name='run_setup_wizard')
def run_setup_wizard():
    """Get user input to create setup file"""
    PIE.run_setup_wizard()


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

@click.command(name='make_position_movie')
@click.argument(
    'xy_pos_idx',
    type=int
)
@click.argument(
    'configfile',
    type=click.Path(file_okay=True, dir_okay=False, exists=True)
)
@click.argument(
    'colony_subset',
    type=str,
    default='growing'
)
def make_pos_movie(xy_pos_idx, configfile, colony_subset):
    """Create a movie of a single position after PIE analysis.

    XY_POS_IDX is an integer corresponding to xy position to be analyzed
    
    CONFIGFILE is the path to configuration setup file for this
    experiment

    COLONY_SUBSET can be:
    -   'growing': to label only those colonies that receive a growth 
        rate measurement after filtration
    -   'tracked': to label all colonies that were tracked, but not 
        include those that were recognized but then removed because
        they were e.g. a minor part of a broken-up colony
    -   'all': to label all colonies detected by PIE
    """
    PIE.make_position_movie(
        xy_pos_idx,
        analysis_config_file = configfile,
        colony_subset = colony_subset)

# -- Create command group -----------------------------------------------------

@click.group()
def cli():  # pragma: no cover
    """Command line interface for PIE."""
    pass


cli.add_command(run_experiment)
cli.add_command(track_single_pos)
cli.add_command(make_pos_movie)
cli.add_command(run_setup_wizard)
