'''
Separate command-line interface to run PIE experiments via flowServ. The
flowServ workflow required the option to override configuration settings via
command-line parameters.

All commands that are defined in this module are available at the console via
command `pieflow`.
'''

import click
import numpy as np

import PIE


# -- Commands -----------------------------------------------------------------

@click.command(name='colony_recognition')
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
@click.argument(
    'input_file',
    type=click.Path(file_okay=True, dir_okay=False, exists=True)
)
@click.argument(
    'output_path',
    type=click.Path(file_okay=False, dir_okay=True)
)
def run_colony_recognition_analysis(
    image_type, hole_fill_area, max_proportion, cleanup, input_file,
    output_path
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
        save_extra_info=True
    )


# -- Create command group -----------------------------------------------------

@click.group()
def cli():  # pragma: no cover
    """Command line interface for PIE."""
    pass


cli.add_command(run_colony_recognition_analysis)
