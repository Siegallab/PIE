PIE Readme
==========

Processing Images Easily (PIE) automatically tracks growing microcolonies in low-resolution brightfield and phase-contrast microscopy images. The program works for recognizing microcolonies in a wide range of microbes, and allows automated measurement of growth rate, lag times, and (if applicable) fluorescence across time for microcolonies founded by single cells. PIE recognizes colony outlines very robustly and accurately across a wide range of image brightnesses and focal depths, and allows simultaneous measurements of growth properties and fluorescence intensities with very high throughput (in our lab, ~100,000 colonies per experiment), including in complex, multiphase experiments.

To learn how to install and use PIE, see the `PIE documentation <https://pie-image.readthedocs.io/en/latest/?>`_.

To learn how PIE works, see `our preprint <https://doi.org/10.1101/253724>`_.

PIE Quickstart
^^^^^^^^^^^^^^

.. quickstart_inclusion

Below is a quick reference to essential PIE functions; see the full documentation for more details.

All the commands below must be run in Terminal (MacOS/Linux) / Command Prompt (Windows)

Installing PIE
--------------

PIE requires **Python 3.6+**, and can be installed using `pip <https://pip.pypa.io/en/stable/>`_, which should come with your python installation.

.. tabs::

    .. tab:: Unix/macOS Terminal

        .. code-block:: bash

            python -m pip install git+https://github.com/Siegallab/PIE

    .. tab:: Windows Command Prompt

        .. code-block:: bash

            py -m pip install git+https://github.com/Siegallab/PIE

See :doc:`Installing PIE <docs/source/installation>` for details.

Analyzing a single image
------------------------

To run PIE on a single image:

.. code-block:: bash

    pie analyze_single_image INPUT_IM_PATH OUTPUT_PATH IMAGE_TYPE

Where:

+ ``INPUT_IM_PATH`` is the path to the image you want to analyze
+ ``OUTPUT_PATH`` is a directory you'd like to store the results of the analysis in
+ ``IMAGE_TYPE`` is 'brightfield' or 'phase_contrast'

See :doc:`Running PIE single-image analysis <docs/source/single_im_analysis>` for details on the inputs and outputs, as well as additional analysis options.

Analyzing timelapse experiments
-------------------------------

To analyze a timelapse experiment, you need to create a setup file containing analysis parameters, and then run the analysis itself.

To interactively create a setup file:

.. code-block:: bash

    pie run_setup_wizard

To analyze the timelapse experiment:

.. code-block:: bash

    pie run_timelapse_analysis PATH_TO_SETUP_FILE

Where ``PATH_TO_SETUP_FILE`` is the path to the setup file created by the setup wizard.

See :doc:`Running PIE timelapse experiments <docs/source/full_experiment>` for information on analyzing complex, multi-phase experiments.

Creating movies
---------------

After timelapose experiments are analyzed, PIE can create movies of the output; this is helpful in understanding whether the analysis worked as expected.

To create simple movies of PIE analysis output for a single imaging position:

.. code-block:: bash

    pie make_position_movie XY_POS PATH_TO_SETUP_FILE

Where:

+ ``XY_POS`` is the imaging position number for which the movie should be created
+ ``PATH_TO_SETUP_FILE`` is the path to the setup file created by the setup wizard

See :doc:`Creating movies of image analysis results <docs/source/movies>` for additional options and examples of more movie types that can be created from PIE output.
