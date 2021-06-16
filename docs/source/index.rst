.. pie documentation master file, created by
   sphinx-quickstart on Sat Apr 24 17:19:46 2021.

Welcome to pie's documentation!
===============================

Processing Images Easily (PIE) automatically tracks growing microcolonies in low-resolution brightfield and phase-contrast microscopy images. The program works for recognizing microcolonies in a wide range of microbes, and allows automated measurement of growth rate, lag times, and (if applicable) fluorescence across time for microcolonies founded by single cells. PIE recognizes colony outlines very robustly and accurately across a wide range of image brightnesses and focal depths, and allows simultaneous measurements of growth properties and fluorescence intensities with very high throughput (in our lab, ~100,000 colonies per experiment), including in complex, multiphase experiments.

See the `PIE preprint <https://doi.org/10.1101/253724>`_ for details on the analysis, and the `PIE github repository <https://github.com/Siegallab/PIE>`_ for the code.

To test microcolony recognition and growth tracking on your images, try our `web application <pie.hpc.nyu.edu>`_.

If you have any questions about setting up or using PIE, we'd love to help! Feel free to contact us at pie-siegal-lab@nyu.edu or `open an issue on github <https://github.com/Siegallab/PIE/issues>`_.

.. toctree::
    :maxdepth: 1

    quickstart

.. toctree::
    :maxdepth: 1
    :caption: Install upgrade and uninstall

    installation
    uninstallation
    
.. toctree::
    :maxdepth: 1
    :caption: Inputs and outputs
    
    inputs
    outputs

.. toctree::
    :maxdepth: 1
    :caption: Running PIE

    single_im_analysis
    full_experiment
    parallelizing

.. toctree::
    :maxdepth: 2
    :caption: Making Movies

    movies

.. toctree::
    :maxdepth: 2
    :caption: Sample Experimental Data

    sample_experiments



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
