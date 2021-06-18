PIE Readme
==========

Processing Images Easily (PIE) automatically tracks growing microcolonies in low-resolution brightfield and phase-contrast microscopy images. The program works for recognizing microcolonies in a wide range of microbes, and allows automated measurement of growth rate, lag times, and (if applicable) fluorescence across time for microcolonies founded by single cells. PIE recognizes colony outlines very robustly and accurately across a wide range of image brightnesses and focal depths, and allows simultaneous measurements of growth properties and fluorescence intensities with very high throughput (in our lab, ~100,000 colonies per experiment), including in complex, multiphase experiments.

To learn how to install and use PIE, see the `PIE documentation <https://pie-image.readthedocs.io/en/latest/?>`_.

To learn how PIE works, see `our preprint <https://doi.org/10.1101/253724>`_.

To test microcolony recognition and growth tracking on your images, try our `web application <pie.hpc.nyu.edu>`_.

PIE Quickstart
^^^^^^^^^^^^^^

.. quickstart_inclusion

Below is a quick reference to essential PIE functions; see the full `PIE documentation <https://pie-image.readthedocs.io/en/latest/?>`_ for more details. If you have any questions about setting up or using PIE, we'd love to help! Feel free to contact us at pie-siegal-lab@nyu.edu or `open an issue on github <https://github.com/Siegallab/PIE/issues>`_.

All the commands below must be run in Terminal (MacOS/Linux) / Command Prompt (Windows)

Installing PIE
--------------

PIE requires **Python 3.6+**, and can be installed using `pip <https://pip.pypa.io/en/stable/>`_, which should come with your python installation.

In unix/macOS Terminal, run:

.. code-block:: bash

    python -m pip install git+https://github.com/Siegallab/PIE@1.0.0

or, in Windows Command Prompt, run:

.. code-block:: bash

    py -m pip install git+https://github.com/Siegallab/PIE@1.0.0

See `Installing PIE <https://pie-image.readthedocs.io/en/latest/installation.html>`_ for details.

Analyzing a single image
------------------------

To run PIE on a single image:

.. code-block:: bash

    pie analyze_single_image INPUT_IM_PATH OUTPUT_PATH IMAGE_TYPE

Where:

+ ``INPUT_IM_PATH`` is the path to the image you want to analyze
+ ``OUTPUT_PATH`` is a directory you'd like to store the results of the analysis in
+ ``IMAGE_TYPE`` is 'bright' (for cells that are brighter than the image background) or 'dark' (for cells that are darker than the image background)

See `Running PIE single-image analysis <https://pie-image.readthedocs.io/en/latest/single_im_analysis.html>`_ for details on the inputs and outputs, as well as additional analysis options.

    .. note::

        If your path contains backslash ('\\') characters (e.g. on Windows) you will need to use a double backslash instead ('\\\\') when specifying the path

Analyzing timelapse experiments
-------------------------------

To analyze a time lapse experiment, you need to create a setup file containing analysis parameters, and then run the analysis itself.

To interactively create a setup file:

.. code-block:: bash

    pie run_setup_wizard

To analyze the timelapse experiment:

.. code-block:: bash

    pie run_timelapse_analysis PATH_TO_SETUP_FILE

Where ``PATH_TO_SETUP_FILE`` is the path to the setup file created by the setup wizard.

    .. note::

    	PIE enforces specific naming conventions for input files in time lapse experiments (e.g. *t01xy0004.tif*). See `Filenames <https://pie-image.readthedocs.io/en/latest/full_experiment.html#filenames>`_ for file naming conventions for time lapse experiments and information on how to encode imaging position, timepoint, and channel during setup.

See `Running PIE timelapse experiments <https://pie-image.readthedocs.io/en/latest/full_experiment.html>`_ for information on analyzing complex, multi-phase experiments.

Creating movies
---------------

After timelapose experiments are analyzed, PIE can create movies of the output; this is helpful in understanding whether the analysis worked as expected.

To create simple movies of PIE analysis output for a single imaging position:

.. code-block:: bash

    pie make_position_movie XY_POS PATH_TO_SETUP_FILE

Where:

+ **XY_POS** is the imaging position number for which the movie should be created (see `Filenames <https://pie-image.readthedocs.io/en/latest/full_experiment.html#filenames>`_ for information on how to encode imaging position in filenames and the setup file)
+ **PATH_TO_SETUP_FILE** is the path to the setup file created by the setup wizard

See `Creating movies of image analysis results <https://pie-image.readthedocs.io/en/latest/movies.html>`_ for additional options and examples of more movie types that can be created from PIE output.
