Sample Experiments
==================

We provide some partial data from three experiments performed in the Siegal lab, and sample setup files that can be used to analyze them.

Downloading example experiments
-------------------------------

If you downloaded the full `PIE github repository <https://github.com/Siegallab/PIE/>`_, you can find the sample experiments in the ``PIE_test_data/IN/`` directory, and the sample setup files in ``sample_PIE_setup_files/``.

If you do not want to download the full github repository, you can use DownGit to selectively download these two folders:

+ `Sample dataset download <https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/Siegallab/PIE/tree/master/PIE_test_data>`_ (~390 Mb)
+ `Sample setup files download <https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/Siegallab/PIE/tree/master/sample_PIE_setup_files>`_

After downloading the folders, you will need to unzip them.

.. note:: The rest of the code assumes that you have PIE installed, and that you are in a directory that contains both the unzipped ``PIE_test_data/`` and ``sample_PIE_setup_files/`` folders.

	Any outputs will be created inside ``PIE_test_data/out/``

Example data
------------

Single-phase growth rate experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Experiment ``PIE_test_data/IN/SL_170619_2_GR_small`` contains a subset of the *Saccharomyces cerevisiae* data described in Figure 5 of the `PIE paper <https://doi.org/10.1101/253724>`_. It is a growth rate experiment that originally had 4032 fields imaged at 10 timepoints; partial images from a small subset of fields is included here. This experiment can be analyzed using either `sample_PIE_setup_files/gr_phase_setup_simple.csv <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files/gr_phase_setup_simple.csv>`_, or `sample_PIE_setup_files/gr_phase_setup.csv <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files/gr_phase_setup.csv>`_; the latter includes advanced options and filtration settings closer to those typically used for analysis in our lab.

.. tabs::

    .. tab:: command-line

        .. code-block:: bash

            pie run_timelapse_analysis sample_PIE_setup_files/gr_phase_setup.csv

    .. tab:: python

        .. code-block:: python

            import PIE
            PIE.run_timelapse_analysis(
                'sample_PIE_setup_files/gr_phase_setup.csv'
                )

Growth rate experiment with post-phase fluorescence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Experiment ``PIE_test_data/IN/EP_170202_small`` contains a subset of the data from a *Saccharomyces cerevisiae* growth rate experiment that originally had 4032 fields imaged at 10 timepoints (located in ``PIE_test_data/IN/EP_170202_small/EP_170202_2_GR``), plus an additional post-growth-phase, fluorescent-only set of images in the GFP channel (located in ``PIE_test_data/IN/EP_170202_small/EP_170202_2_FC``). Each imaging position includes cells from two strains, one of which is expressing GFP driven by the Scw11 promoter; GFP levels can therefore be used to differentiate between these strains, and setting the first phase of the experiment as the 'linked' phase for the fluorescence imaging will automatically trigger fluorescence-based classification of each colony.

Note also that here, the brightfield images are saved as ``.jpg`` files; while not ideal, ``PIE`` can still process these images.

This experiment can be analyzed using either `sample_PIE_setup_files/gr_with_postfluor_setup_simple.csv <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files/gr_with_postfluor_setup_simple.csv>`_ or `sample_PIE_setup_files/gr_with_postfluor_setup.csv <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files/gr_with_postfluor_setup.csv>`_; the latter includes advanced options and filtration settings closer to those typically used for analysis in our lab.

.. tabs::

    .. tab:: command-line

        .. code-block:: bash

            pie run_timelapse_analysis sample_PIE_setup_files/gr_with_postfluor_setup.csv

    .. tab:: python

        .. code-block:: python

            import PIE
            PIE.run_timelapse_analysis(
                'sample_PIE_setup_files/gr_with_postfluor_setup.csv'
                )

Two-phase growth rate experiment with simultaneous fluorescence measurement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Experiment ``PIE_test_data/IN/SL_180519_small`` contains a subset of the data from a *Saccharomyces cerevisiae* growth rate experiment from Figure 9 in `Li *et al.* 2018 <https://doi.org/10.1371/journal.pgen.1007744>`_. The experiment initially consisted of 1200 imaging fields that were imaged over two phases before and after a heat shock: five timepoints of pre-heat shock growth rate (located in ``PIE_test_data/IN/SL_180519_small/SL_180519_2_GR``) and fifteen timepoints post-heat shock (located in ``PIE_test_data/IN/SL_180519_small/SL_180519_2_AT``). Fluorescent data was also collected at each timepoint during both phases: detecting MitoTracker staining and Tsl1-GFP expression in the first phase, and only Tsl1-GFP in the second phase.

An analysis of just the first phase of the experiment (including fluorescence data) can be used to explore the output of PIE for combined growth/fluorecence experiments using `sample_PIE_setup_files/gr_with_fluor_setup_simple.csv <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files/gr_with_fluor_setup_simple.csv>`_

.. tabs::

    .. tab:: command-line

        .. code-block:: bash

            pie run_timelapse_analysis sample_PIE_setup_files/gr_with_fluor_setup_simple.csv

    .. tab:: python

        .. code-block:: python

            import PIE
            PIE.run_timelapse_analysis(
                'sample_PIE_setup_files/gr_with_fluor_setup_simple.csv'
                )

The full experiment can be analyzed using either `sample_PIE_setup_files/two_phase_setup_simple.csv <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files/two_phase_setup_simple.csv>`_, or `sample_PIE_setup_files/two_phase_setup.csv <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files/two_phase_setup.csv>`_; the latter includes advanced options and filtration settings closer to those typically used for analysis in our lab. Note that these setup files are set up to exclude images for the first two time points after heat shock from the analysis, as heat-induced warping causes colonies to be out of focus during these time points.

.. tabs::

    .. tab:: command-line

        .. code-block:: bash

            pie run_timelapse_analysis sample_PIE_setup_files/two_phase_setup.csv

    .. tab:: python

        .. code-block:: python

            import PIE
            PIE.run_timelapse_analysis(
                'sample_PIE_setup_files/two_phase_setup.csv'
                )
