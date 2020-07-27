'''
Simple command-line interface to run PIE experiment.
'''

import click
import numpy as np

import PIE


# -- Commands -----------------------------------------------------------------

@click.command(name='config')
@click.argument(
    'configfile',
    type=click.Path(file_okay=True, dir_okay=False, exists=True)
)
def run_experiment(configfile):
    """Run a full image analysis experiment."""
    PIE.run_growth_rate_analysis(configfile)


@click.command(name='experiment')
@click.option(
    '-t', '--total_timepoint_num',
    required=True,
    type=click.INT,
    help='Highest imaged timepoint number (default=10).'
)
@click.argument(
    'input_path',
    type=click.Path(file_okay=False, dir_okay=True, exists=True)
)
@click.argument(
    'output_path',
    type=click.Path(file_okay=False, dir_okay=True)
)
def run_multi_image_experiments(total_timepoint_num, input_path, output_path):
    """Run a full image analysis experiment."""
    PIE.run_default_growth_rate_analysis(
        input_path=input_path,
        output_path=output_path,
        total_timepoint_num=total_timepoint_num
    )


@click.command(name='single')
@click.option(
    '-t', '--image_type',
    required=True,
    help=(
        'The type of micrscopy used to generate images (either \'brightfield\''
        ' or \'phasecontrast\').'
    )
)
@click.option(
    '-h', '--hole_fill_area',
    required=False,
    default=None,
    help=(
        'Size (in pixels) of the largest empty space between cells (on the '
        'inside of a colony) to consider part of the colony (default=inf).'
    )
)
@click.option(
    '-m', '--max_proportion',
    required=False,
    default=0.75,
    type=click.FLOAT,
    help=(
        'The max proportion of the edge of a detected colony \'pie piece\' '
        'that is allowed to not touch a neighboring pie piece before being '
        'removed during cleanup step. Recommended to be set to 0.25-0.4. Only '
        'applies during cleanup steps (default=0.75).'
    )
)
@click.option(
    '-c', '--cleanup',
    is_flag=True,
    default=False,
    help=(
        'Remove pieces of background detected as part of the colony '
        '(default=False).'
    )
)
@click.option(
    '-e', '--save_extras',
    is_flag=True,
    default=False,
    help='Save additional output files (default=False).'
)
@click.argument(
    'input_file',
    type=click.Path(file_okay=True, dir_okay=False, exists=True)
)
@click.argument(
    'output_path',
    type=click.Path(file_okay=False, dir_okay=True)
)
def run_single_image_analysis(
    image_type, hole_fill_area, max_proportion, cleanup, save_extras,
    input_file, output_path
):
    """Analyze a single image using PIE."""
    if hole_fill_area is None or hole_fill_area.lower() == 'inf':
        hole_fill_area = np.inf
    else:
        hole_fill_area = int(hole_fill_area)
    PIE.analyze_single_image(
        input_im_path=input_file,
        output_path=output_path,
        image_type=image_type,
        hole_fill_area=hole_fill_area,
        cleanup=cleanup,
        max_proportion_exposed_edge=max_proportion,
        save_extra_info=save_extras
    )


# -- Create command group -----------------------------------------------------

@click.group(name='run')
def runcli():
    """Run image analysis."""
    pass


runcli.add_command(run_multi_image_experiments)
runcli.add_command(run_single_image_analysis)
runcli.add_command(run_experiment)


@click.group()
def cli():  # pragma: no cover
    """Command line interface for PIE."""
    pass


cli.add_command(runcli)
