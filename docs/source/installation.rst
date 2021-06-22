Install and upgrade PIE
=======================

Requirements
------------

PIE requires **Python 3.6+**; you will also need the following python packages, but they will be automatically installed by following the instructions below.

+ numpy ≥1.16.0
+ opencv-contrib-python >4.0.0.0
+ pandas ≥1.1.0
+ Pillow ≥7.2.0
+ scipy ≥1.1.0
+ plotnine ≥ 0.7.1
+ pyarrow
+ Click ≥7.0
+ scikit-learn ≥0.23.0

Installation
------------

PIE can be installed using `pip <https://pip.pypa.io/en/stable/>`_, which should come with your python installation.

.. tabs::

    .. tab:: Unix/macOS 

        If you are using a Linux or MacOS operating system, python comes pre-installed on your computer. All you need to do to install the latest stable version of the PIE package is open the Terminal application, paste the following line, and press 'enter'.

        .. code-block:: bash

            python -m pip install git+https://github.com/Siegallab/PIE@v1.0.1

        .. warning::

            OpenCV, one of the key packages required for PIE, cannot yet be automatically installed on computers with Apple M1 CPUs. If your computer has an Apple M1 CPU, you will need to first build OpenCV on your computer following `these instructions <https://sayak.dev/install-opencv-m1/>`_, **before installing PIE** as described above. We expect this issue to be resolved in the near future.

    .. tab:: Windows

        If you do not already have python installed on your computer, you can download the latest version for free from the Python website: https://www.python.org/downloads/

        .. note::

            In order to be able to complete the PIE installation steps below, please make sure that you select "Add Python to PATH" during installation.

        Once you have python installed, open the ``cmd`` program from the Start Menu, and type the following to install the latest stable version of the PIE package:

        .. code-block:: bash

            py -m pip install git+https://github.com/Siegallab/PIE@v1.0.1

        .. admonition:: Troubleshooting
            :class: dropdown, tip

            If you see either:

            #. A warning telling you the script is not installed in your PATH, or
            #. ``py is not recognized as an internal or external command``,
            
            then Windows likely does not know where to find the python program. You have to resolve this issue before proceeding with installation; see `here <https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/>`_ for directions on how to do this.

            Depending on your version of Windows, you may see ``WARNING: Couldn't compute FAST_CWD pointer`` and/or ``cygwin_warning``, with additional text, during installation; this may be followed by a period of a few minutes when nothing else changes on the screen. These warnings can safely be ignored, and package installation will continue automatically.

Upgrade
-------

For now, to upgrade PIE to the most current version, you will have to uninstall PIE first and reinstall. Directly reinstall PIE without uninstall may not include all the upgrades. Please refer to this section and the :doc:`uninstall PIE <uninstallation>` section for install and uninstall guidelines. 
