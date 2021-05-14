PIE Inputs
==========

Imaging conditions
------------------

PIE can segment colonies from brightfield or phase-contrast images, and can use fluorescent images to calculate fluorescence levels within segmented colonies.

For **brightfield imaging**, ideal images for PIE analysis are slightly defocused, with dark outlining surrounding cells that are brighter than the background. For **phase-contrast imaging**, a wide range of imaging conditions are acceptable. See figures 1-4 in the `PIE preprint <https://doi.org/10.1101/253724>`_ for example images. Importantly, PIE performs well on low-resolution images.

Setup files
-----------

Time-lapse PIE experiments require a single *csv*-format setup file to run. Some templates for experiment types commonly run in the Siegal lab can be found in `sample_PIE_setup_files <https://github.com/Siegallab/PIE/blob/master/sample_PIE_setup_files>`_. See the :ref:`Setting up time-lapse experiments <setting up timelapse>` section for more details on how to create new setup files and the parameters that need to be set.

File naming conventions
-----------------------

Naming of single analyzed image files can be arbitrary.

For time-lapse experiments, file names may contain a label for the imaging position, fluorescence channel, and/or timepoint, in any order or combination; however, the naming convention must be coded into the setup file. Read more about how to encode file-naming conventions into your setup file :ref:`here <filename_convention>`.