'''
Simple command-line interface to run PIE experiment.
'''

import click
import numpy as np

import PIE

# -- Commands -----------------------------------------------------------------

@click.command('run_timelapse_analysis')
@click.argument(
    'configfile',
    type=click.Path(file_okay=True, dir_okay=False, exists=True)
    )
@click.option(
    '-r', '--repeat_image_analysis_and_tracking', default=False, type=bool
    )
def run_timelapse_analysis(configfile, repeat_image_analysis_and_tracking):
    """Run a full image analysis experiment."""
    PIE.run_timelapse_analysis(
        analysis_config_file = configfile,
        repeat_image_analysis_and_tracking = repeat_image_analysis_and_tracking
        )

@click.command(name='analyze_single_image')
@click.argument(
    'input_im_path',
    type=click.Path(file_okay=True, dir_okay=False, exists=True)
    )
@click.argument(
    'output_path',
    type=click.Path(file_okay=False, dir_okay=True)
    )
@click.argument(
    'image_type',
    type=click.Choice(['bright','dark'], case_sensitive=False)
    )
@click.option(
    '-h',
    '--hole_fill_area',
    default=np.inf,
    type=float,
    )
@click.option(
    '-c',
    '--cleanup',
    default=False,
    type=bool
    )
@click.option(
    '-m',
    '--max_proportion_exposed_edge',
    default=0.75,
    type=click.FloatRange(min=0, max=1)
    )
@click.option(
    '-i',
    '--cell_intensity_num',
    default=1,
    type=click.IntRange(min=1, max=2)
    )
@click.option(
    '-s', '--save_extra_info', default=True, type=bool
    )
def analyze_single_image(
    input_im_path,
    output_path,
    image_type,
    hole_fill_area,
    cleanup,
    max_proportion_exposed_edge,
    cell_intensity_num,
    save_extra_info
    ):
    """Run a full image analysis experiment."""
    PIE.analyze_single_image(
        input_im_path,
        output_path,
        image_type,
        hole_fill_area = hole_fill_area,
        cleanup = cleanup,
        max_proportion_exposed_edge = max_proportion_exposed_edge,
        cell_intensity_num = cell_intensity_num,
        save_extra_info = save_extra_info
        )


@click.command(name='run_setup_wizard')
def run_setup_wizard():
    """Get user input to create setup file"""
    PIE.run_setup_wizard()


@click.command(name='track_single_pos')
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
@click.option(
    '-s', '--colony_subset',
    type=click.Choice(['growing','tracked','all'], case_sensitive=False),
    default='growing'
)
@click.option(
    '-m', '--movie_format',
    type=click.Choice(
        ['jpeg','jpg','tiff','tif','gif','h264','mjpg','mjpeg'],
        case_sensitive=False
        ),
    default='gif'
)
def make_position_movie(xy_pos_idx, configfile, colony_subset, movie_format):
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
        colony_subset = colony_subset,
        movie_format = movie_format)

# -- Create command group -----------------------------------------------------

@click.group()
def cli():  # pragma: no cover
    """Command line interface for PIE."""
    pass

cli.add_command(analyze_single_image)
cli.add_command(run_timelapse_analysis)
cli.add_command(track_single_pos)
cli.add_command(make_position_movie)
cli.add_command(run_setup_wizard)
