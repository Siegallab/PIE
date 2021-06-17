Parallelizing PIE analysis
==========================

Although PIE does not automatically support multithreading, it is set up to easily be run via batch submission. To do this, the colony tracking step for each individual xy position is separated from the steps that compile the data across positions and calculate growth rates.

Analyzing individual imaged xy positions
----------------------------------------

To analyze each imaged position individually (e.g. via batch jobs), you need to pass the integer corresponding to the xy position being analyzed and the path to the setup file. This can be done e.g. for position 11 of the ``SL_170619_2_GR_small`` sample data:

.. tabs::

    .. tab:: command-line

        .. code-block:: bash

            pie track_single_pos 11 /local/path/to/PIE/sample_PIE_setup_files/gr_phase_setup.csv

    .. tab:: python

        .. code-block:: python

            import PIE
            PIE.track_colonies_single_pos(
                11,
                '/local/path/to/PIE/sample_PIE_setup_files/gr_phase_setup.csv'
                )

Combining position-wise data and calculating growth rates
---------------------------------------------------------

After every imaged xy position has been analyzed, the data can be combined and growth rates can be calculated by simply running ``PIE.run_timelapse_analysis(...)`` (or ``pie run_timelapse_analysis ...``), as in :doc:`full_experiment`, with the default ``repeat_image_analysis_and_tracking=False`` option.