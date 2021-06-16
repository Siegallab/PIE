Running PIE timelapse experiments
===================================

.. _setting up timelapse:

Setting up timelapse experiments
----------------------------------

New setup files can also be created using the ``run_setup_wizard`` command, which walks the user through all the key parameters to enter one-by-one. This is recommended for users running new experiment types for the first time.

.. tabs::

    .. tab:: command-line

        .. code-block:: bash

            pie run_setup_wizard

Time-lapse PIE experiments require a single *csv*-format setup file to run. Some templates for experiment types commonly run in the Siegal lab can be found in `sample_PIE_setup_files <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files>`_. Details on each setup file are provided below, and each can be used to analyze an experiment provided in the `PIE_test_data/IN <https://github.com/Siegallab/PIE/blob/master/PIE_test_data/IN>`_ folder on github (see :doc:`Sample Experiments <sample_experiments>`).

General experimental setup
^^^^^^^^^^^^^^^^^^^^^^^^^^

To set up a file for an experiment, modify an existing setup file. Setup files must be *.csv* files, and must have at least a 'Parameter' column for the name of the parameter being set, and a 'Value' column for its value.

The following are the minimal required parameters, and must be included in each experiment:

.. csv-filter:: Required Setup File Parameters
   :file: ../../PIE/PIE_data/param_descriptions.csv
   :included_cols: 0,3
   :widths: 1, 3
   :include: {1: 'required'}
   :width: 100%
   :header-rows: 1
   :stub-columns: 1

The PIE github repository contains an example `minimal setup file <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files/gr_phase_setup_simple.csv>`_.

.. _filename_convention:

Filenames
^^^^^^^^^

Note that the values of many of these parameters determine the format of the filenames; see examples below for sample filenames and corresponding parameter values. Note that the number of digits in the time point and position is dependent on ``total_time point_num`` and ``max_xy_position_num``, respectively.

+----------------------------+---------------------------------------------------------+
|                            | filename                                                |
|                            +----------------------------+----------------------------+
| Parameter                  | *t1xy001c3.tif*            | *pos1tp01.jpg*             |
+============================+============================+============================+
| **label_order_list**       | timepoint;position;channel | position;timepoint         |
+----------------------------+----------------------------+----------------------------+
| **position_label_prefix**  | xy                         | pos                        |
+----------------------------+----------------------------+----------------------------+
| **timepoint_label_prefix** | t                          | tp                         |
+----------------------------+----------------------------+----------------------------+
| **main_channel_label**     | c3                         |                            |
+----------------------------+----------------------------+----------------------------+
| **max_xy_position_num**    | 999                        | 9                          |
+----------------------------+----------------------------+----------------------------+
| **max_timepoint_num**      | 9                          | 99                         |
+----------------------------+----------------------------+----------------------------+
| **im_file_extension**      | tif                        | jpg                        |
+----------------------------+----------------------------+----------------------------+

.. tip::
    The reliance on **max_xy_position_num** and **max_timepoint_num** to set the number of digits in the position and timepoint of the filename can be problematic for analysis of data that includes only a subset of the originally collected images. For example, if a user initially collected 100 timepoints worth of data, filenames are most likely t001-t100. However, if the user wants to wants to only analyze the first 99 of these timepoints and sets **max_timepoint_num** to 99, PIE will expect only two digits in the timepoint (t01-t99). This issue can be circumvented by adding any extra 0s to **timepoint_label_prefix**, i.e. setting it to 't0' instead of 't' in this example.

Running the experiment
----------------------

To run a full image analysis experiment using PIE, you can use the ``run_timelapse_analysis`` function, which takes a setup file path as input:

.. tabs::

    .. tab:: python

        .. code-block:: python

            import PIE
            PIE.run_timelapse_analysis(
                '/full/path/to/setup_file.csv'
                )

    .. tab:: command-line

        .. code-block:: bash

            pie run_timelapse_analysis /full/path/to/setup_file.csv

.. note:: Replace ``/full/path/to/setup_file.csv`` in the code above with the full path to the setup file for your experiment; also, see :doc:`Sample Experiments <sample_experiments>`. For file path format of Windows, MacOS or Unix, please refer to examples in :doc:`Running PIE single-image analysis <single_im_analysis>`. 

Although many modifications to experiment setup and analysis can be made (see below), these changes are achieved by altering the setup file; all experiments can then be run using the code above.

Advanced analysis options
-------------------------

In addition to the default experiment processing parameters, a number of optional parameters can be altered that affect file processing, image analysis/colony recognition, and filtration of growth rates:

Additional processing options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following additional options pertain to processing of your image files:

.. csv-filter:: Additional general setup file parameters
   :file: ../../PIE/PIE_data/param_descriptions.csv
   :included_cols: 0,2,3
   :widths: 2,1,6
   :include: {1: 'general'}
   :width: 100%
   :header-rows: 1
   :stub-columns: 1

Modifying image analysis
^^^^^^^^^^^^^^^^^^^^^^^^

The following optional parameters allow users to modify how image analysis is performed:

.. csv-filter:: Setup file parameters pertaining to image analysis
   :file: ../../PIE/PIE_data/param_descriptions.csv
   :included_cols: 0,2,3
   :widths: 2,1,6
   :include: {1: 'image analysis'}
   :width: 100%
   :header-rows: 1
   :stub-columns: 1

Modifying growth rate filtration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following optional parameters allow users to modify how growth rates measured for a time series experiment are filtered:

.. csv-filter:: Setup file parameters pertaining to growth rate filtration
   :file: ../../PIE/PIE_data/param_descriptions.csv
   :included_cols: 0,2,3
   :widths: 2,1,6
   :include: {1: 'growth rate filtration'}
   :width: 100%
   :header-rows: 1
   :stub-columns: 1

The PIE github repository contains an example `setup file with non-default parameter values <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files/gr_phase_setup_simple.csv>`_.

Adding fluorescent measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In experiments in which fluorescence data is collected alonside brightfield/phase contrast data, additional parameters must be provided in order for PIE to process the fluorescence data; note that all the default parameters here are empty, which results in a default of no fluorescence analysis being performed:

.. csv-filter:: Setup file parameters pertaining to fluorescence measurements
   :file: ../../PIE/PIE_data/param_descriptions.csv
   :included_cols: 0,2,3
   :widths: 2,1,6
   :include: {1: 'fluorescence measurements'}
   :width: 100%
   :header-rows: 1
   :stub-columns: 1

Colony outlines are always calculated based on a "main channel", which should consist of either brightfield or phase contrast images; the colonies identified in the main channel will then be overlaid on any fluorescent images in the phase to calculate fluorescence levels.

We provide an example `setup file with fluorescence data analysis <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files/gr_with_fluor_setup_simple.csv>`_ in the PIE github repository.

Analysis of complex experiments
-------------------------------

.. _phase explanation:

Phases
^^^^^^

Each experiment may consist of one or more phases. A single phase consists of a single, continuous bout of imaging. PIE can analyze experiments consisting of multiple such phases. During growth rate analysis, growth rates will be calculated independently for any phase that contains multiple time points, but colony identities will be linked across phases. Multi-phase experiments are meant to allow users to continue to track the same colonies across multiple experimental treatments, with growth rate and lag reported independently for each.

To specify parameters for multiple experimental phases, add a ``PhaseNum`` column to your setup file. Phases must be consecutive integers (i.e. '1', '2', etc). For any parameters that differ between phases, the parameter must be specified for each phase on an individual line with its corresponding ``PhaseNum``. For parameters that are common between experimental phases (e.g. ``output_path``), PhaseNum may be set to 'all'.

Because each phase of a multi-phase experiment should be imaged with the same set of imaging positions, and the outputs of all phases are collected in a single output folder, the values of the following parameters must be the same across all phases:

+ ``output_path``
+ ``im_format``
+ ``first_xy_position``
+ ``max_xy_position_num``
+ ``extended_display_positions``.

We provide an example `two-phase setup file with fluorescence data analysis <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files/two_phase_setup_simple.csv>`_ in the PIE github repository.

Post-phase fluorescent measurement and fluorescence-based classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For experiments in which fluorescent and non-fluorescent strains (or strains with different fluorescent markers) are co-cultured, PIE can use fluorescence data to classify colonies by strain. It can be useful to collect this kind of 'classification' fluorescence data after an experiment (or experimental phase) is complete, to avoid spending time imaging in a fluorescent channel between each set of time points. Colony segmentation from brightfield or phase-contrast imaging in the previous phase can then be used to assign fluorescent values to colonies.

PIE allows for the creation of a special phase that includes only fluorescent images for a single time point, in which case the ``linked_phase`` parameter should be set to the phase number of the phase containing the brightfield/phase contrast data to be used for colony segmentation (we strongly recommend that this be the phase immediately before the fluorescent classification phase).

These 'post-phase fluorescence' phases require only a subset of parameters to be specified:

+ ``linked_phase``
+ ``fluor_channel_scope_labels``
+ ``fluor_channel_names``
+ ``fluor_channel_thresholds``
+ ``fluor_channel_time points``
+ ``input_path``
+ ``first_xy_position``
+ ``extended_display_positions``
+ ``time point_label_prefix``
+ ``output_path``
+ ``im_file_extension``
+ ``label_order_list``
+ ``max_xy_position_num``
+ ``position_label_prefix``
+ ``im_format``

We provide an example `post-phase fluorescence setup file <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files/gr_with_postfluor_setup_simple.csv>`_ in the PIE github repository.

