Running PIE multi-image experiments
===================================

Setting up multi-image experiments
----------------------------------

Complete PIE experiments require a single *csv*-format setup file to set up. Some templates for experiment types commonly run in the Siegal lab can be found in `sample_PIE_setup_files <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files>`_. Details on each setup file are provided below, and each can be used to analyze an experiment provided in the `PIE_test_data/IN <https://github.com/Siegallab/PIE/blob/master/PIE_test_data/IN>`_ folder on github.

General experimental setup
^^^^^^^^^^^^^^^^^^^^^^^^^^

To set up a file for an experiment, modify an existing setup file. Setup files are *.csv* files that have at least a 'Parameter' column for the name of the parameter being set, and a 'Value' column for its value.

The following are the minimal required parameters, and must be included in each experiment:

.. csv-table:: Required Setup File Parameters
   :file: _static/setup_tables/required_params.csv
   :widths: 25, 75
   :header-rows: 1
   :stub-columns: 1

An example `minimal setup file <https://github.com/Siegallab/PIE/blob/doc_update/sample_PIE_setup_files/gr_phase_setup_simple.csv>`_.

Filenames
*********

Note that the values of many of these parameters determine the format of the filenames; see examples below for sample filenames and corresponding parameter values. Note that the number of digits in the timepoint and position is dependent on ``total_timepoint_num`` and ``total_xy_position_num``, respectively.

+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                   |                                                                              Parameter                                                                             |
|   filename        +----------------------------+-----------------------+------------------------+--------------------+-----------------------+---------------------+-------------------+
|                   |      label_order_list      | position_label_prefix | timepoint_label_prefix | main_channel_label | total_xy_position_num | total_timepoint_num | im_file_extension |
+===================+============================+=======================+========================+====================+=======================+=====================+===================+
| ``t1xy001c3.tif`` | timepoint;position;channel | xy                    | t                      | c3                 | 300                   | 5                   | tif               |
+-------------------+----------------------------+-----------------------+------------------------+--------------------+-----------------------+---------------------+-------------------+
| ``pos1tp01.jpg``  | position;timepoint;channel | pos                   | tp                     |                    | 4                     | 99                  | jpg               |
+-------------------+----------------------------+-----------------------+------------------------+--------------------+-----------------------+---------------------+-------------------+

Running the experiment
----------------------

To run a full image analysis experiment using PIE, you can use the ``run_growth_rate_analysis`` function, which takes a setup file path as input:

.. tabs::

    .. tab:: python

        .. code-block:: python

            import PIE
            PIE.run_growth_rate_analysis(
                analysis_config_file =
                '/full/path/to/setup_file.csv'
                )

    .. tab:: command-line

        .. code-block:: bash

            pie run /full/path/to/setup_file.csv

.. note:: Replace ``/full/path/to/setup_file.csv`` in the code above with the full path to the setup file for your experiment; also, see :doc:`Sample Experiments <sample_experiments>`

Although many modifications to experiment setup and analysis can be made (see below), these changes are achieved by altering the setup file; all experiments can then be run using the code above.

Advanced analysis options
-------------------------

In addition to the default experiment processing parameters, a number of optional parameters can be altered that affect file processing, image analysis/colony recognition, and filtration of growth rates:

Additional processing options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following additional options pertain to processing of your image files:

.. csv-table:: Processing Parameters
   :file: _static/setup_tables/processing_params.csv
   :widths: 25, 10, 65
   :header-rows: 1
   :stub-columns: 1

Modifying image analysis
^^^^^^^^^^^^^^^^^^^^^^^^

The following optional parameters allow users to modify how image analysis is performed:

.. csv-table:: Image Analysis Parameters
   :file: _static/setup_tables/imaging_params.csv
   :widths: 25, 10, 65
   :header-rows: 1
   :stub-columns: 1

Modifying growth rate filtration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following optional parameters allow users to modify how growth rates measured for a time series experiment are filtered:

.. csv-table:: Growth Rate Filtration Parameters
   :file: _static/setup_tables/growth_params.csv
   :widths: 25, 10, 65
   :header-rows: 1
   :stub-columns: 1

We provide an example `setup file with non-default parameter values <https://github.com/Siegallab/PIE/blob/doc_update/sample_PIE_setup_files/gr_phase_setup_simple.csv>`_.

Adding fluorescent measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In experiments in which fluorescence data is collected alonside brightfield/phase contrast data, additional parameters must be provided in order for PIE to process the fluorescence data:

.. csv-table:: Fluorescence Parameters
   :file: _static/setup_tables/fluor_params.csv
   :widths: 25, 10, 65
   :header-rows: 1
   :stub-columns: 1

Colony outlines are always calculated based on a "main channel", which should consist of either brightfield or phase contrast images; the colonies identified in the main channel will then be overlaid on any fluorescent images in the phase to calculate fluorescence levels.

We provide an example `setup file with fluorescence data analysis <https://github.com/Siegallab/PIE/blob/doc_update/sample_PIE_setup_files/gr_with_fluor_setup_simple.csv>`_ here.

Analysis of complex experiments
-------------------------------

Phases
^^^^^^

Each experiment may consist of one or more phases. A single phase consists of a single, continuous bout of imaging. PIE can analyze experiments consisting of multiple such phases. During growth rate analysis, growth rates will be calculated independently for any phase that contains multiple timepoints, but colony identities will be linked across phases. Multi-phase experiments are meant to allow users to continue to track the same colonies across multiple experimental treatments, with growth rate and lag reported independently for each.

To specify parameters for multiple experimental phases, add a ``PhaseNum`` column to your setup file. Phases must be consecutive integers (i.e. '1', '2', etc). For any parameters that differ between phases, the parameter must be specified for each phase on an individual line with its corresponding ``PhaseNum``. For parameters that are common between experimental phases (e.g. ``output_path``), PhaseNum may be set to 'all'.

Because each phase of a multi-phase experiment should be imaged with the same set of imaging positions, and the outputs of all phases are collected in a single output folder, the values of the following parameters must be the same across all phases: ``output_path``, ``im_format``, ``first_xy_position``, ``total_xy_position_num``, and ``extended_display_positions``.

We provide an example `two-phase setup file with fluorescence data analysis <https://github.com/Siegallab/PIE/blob/doc_update/sample_PIE_setup_files/two_phase_setup_simple.csv>`_.

Post-phase fluorescent measurement and fluorescence-based classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PIE can use fluorescence data to classify colonies co-cultured in a single well. It can be useful to collect this kind of 'classification' fluorescence data after an experiment (or experimental phase) is complete, to avoid spending time imaging images in a fluorescent channel between each set of timepoints. Colony segmentation from brightfield or phase-contrast imaging in the previous phase can then be used to assign fluorescent values to colonies.

PIE allows for the creation of a special phase that includes only fluorescent images for a single timepoint, in which case the ``parent_phase`` parameter should be set to the phase number of the phase containing the brightfield/phase contrast data to be used for colony segmentation.

These 'post-phase fluorescence' phases require only a subset of parameters to be specified: ``fluor_channel_scope_labels``, ``fluor_channel_names``, ``fluor_channel_thresholds``, ``fluor_channel_timepoints``, ``input_path``, ``first_xy_position``, ``extended_display_positions``, ``timepoint_label_prefix``, ``output_path``, ``im_file_extension``, ``label_order_list``, ``total_xy_position_num``, ``position_label_prefix``, ``im_format``, and ``parent_phase``.

We provide an example `post-phase fluorescence setup file <https://github.com/Siegallab/PIE/blob/doc_update/sample_PIE_setup_files/gr_with_postfluor_setup_simple.csv>`_.

