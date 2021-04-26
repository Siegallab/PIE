Running PIE multi-image experiments
===================================

Setting up multi-image experiments
----------------------------------

Complete PIE experiments require a single *csv*-format setup file to set up. Some templates for experiment types commonly run in the Siegal lab can be found in `sample_PIE_setup_files <https://github.com/Siegallab/PIE/sample_PIE_setup_files>`_.

General setup
^^^^^^^^^^^^^

To set up a file for an experiment, modify an existing setup file (or make your own from scratch, although this is not recommended!) More detailed explanations of all the parameters will be added later, but for now, you can just read the *Explanation* column for each parameter in the setup files provided.

Phases
^^^^^^

Each experiment may consist of one or more phases; the list of phases in the experiment must be provided in the experimental setup file. A single phase consists of a single, continuously labeled bout of imaging. Colony outlines are always calculated based on a "main channel", which should consist of either brightfield or phase contrast images; the colonies identified in the main channel will then be overlaid on any fluorescent images in the phase to calculate fluorescence levels.

A phase can also be run that takes only fluorescent images for a single timepoint, in which case ``parent_phase`` should be set to the phase number of the phase containing the brightfield/phase contrast data corresponding to the fluorescent images (see `Setup file Examples`_ below).

During growth rate analysis, growth rates will be calculated independently for any phase that contains multiple timepoints, but colony identities will be linked across phases.

Setup file Examples
^^^^^^^^^^^^^^^^^^^

Here are some examples of setup files for experiments commonly done in the Siegal Lab; each corresponds to a single test data folder located in `PIE_test_data/IN/ <https://github.com/Siegallab/PIE/PIE_test_data/IN>`_:

+ `setup file <https://github.com/Siegallab/PIE/sample_PIE_setup_files/gr_phase_setup.csv>`_ for an experiment consisting of only growth rate measurements
+ `setup file <https://github.com/Siegallab/PIE/sample_PIE_setup_files/gr_with_postfluor_setup.csv>`_ for an experiment consisting of growth rate measurements, followed by by a single timepoint of post-phase fluorescent imaging
+ `setup file <https://github.com/Siegallab/PIE/sample_PIE_setup_files/two_phase_setup.csv>`_ for an experiment consisting of two phases of growth rate measurements, the first with two fluorescent channels, the second with a single fluorescent channel

Running the experiment
----------------------

To run a full image analysis experiment using PIE, you can use the ``run_growth_rate_analysis`` function, which takes a setup file path as input:

.. tabs::

    .. tab:: python

        .. code-block:: python

            import PIE
            PIE.run_growth_rate_analysis(
                analysis_config_file =
                '/local/path/to/PIE/sample_PIE_setup_files/gr_phase_setup.csv'
                )

    .. tab:: command-line

        .. code-block:: bash

            pie run /local/path/to/PIE/sample_PIE_setup_files/gr_phase_setup.csv
