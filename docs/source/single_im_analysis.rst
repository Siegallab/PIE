Running PIE single-image analysis
=================================

To analyze a single image using PIE, you can use the ``analyze_single_image`` function.

Inputs:
^^^^^^^

+ **input_im_path**: *{path}*
    the path to the image to be analyzed
+ **output_path**: *{path}*
    the directory in which image analysis folders should be created
+ **image_type**: *{'bright' or 'dark'}*
    The brightness of the cell bodies relative to the image background.
+ **hole_fill_area**: *{integer or 'inf'}, optional, default 'inf'*
    the area (in pixels) of the largest size hole to fill in colony masks after image analysis. For low-resolution yeast imaging, we recommend setting this value to 'inf' (i.e. all the holes in the colonies get filled)
+ **cleanup**: *{True or False} optional, default False*
    whether or not to perform 'cleanup' of spurious pieces of background attached to colonies. (We recommend trying PIE both with and without cleanup on a set of sample images; you can see the Li, Plavskin *et al.* paper for details)
+ **max_proportion_exposed_edge**: *{0-1} optional, default 1*
    maximum proportion of the perimeter of a PIE-detected gradient object ('PIE piece') that may be non-adjacent to another PIE piece to avoid being removed during 'cleanup' steps; only used if **cleanup** is *True*. 0.75 works well for 10x yeast images, and many other images tried
+ **cell_intensity_num**: *{1 or 2} optional, default 1*
    number of distinct intensity categories belonging to cells in the image; determines whether 2 or 3 Gaussians are used to fit intensity log histogram for threshold determination. Can be set to 1 (default) or (in rare cases) 2.
+ **save_extra_info**: *{True or False} optional, default True*
    whether to write additional files described above

.. tabs::

    .. tab:: python

        .. code-block:: python

            import PIE
            colony_mask, colony_property_df = \
                PIE.analyze_single_image(
                    input_im_path,
                    output_path,
                    image_type,
                    hole_fill_area = 'inf',
                    cleanup = False,
                    max_proportion_exposed_edge = 0.75,
                    cell_intensity_num = 1,
                    save_extra_info = True
                    )

        The function returns:

        + **colony_mask**:
            a numpy boolean matrix with True at positions corresponding to pixels in the original image where a colony was detected
        + **colony_property_df**:
            a pandas dataframe containing the properties of every colony in the image (same as the ones saved to *single_image_colony_centers* below, but also containing a list of all the pixels in which each colony was detected).

        .. note::

            Inputs besides **input_im_path**, **output_path**, and **image_type** are optional.

        .. admonition:: Examples
            :class: dropdown, tip

            Analyze an image with bright cells on a dark background, *E:\\trial\\my_image.tif*, and save outputs in a new folder called *E:\\my output folder*

            .. code-block:: python

                import PIE
                colony_mask, colony_property_df = \
                    PIE.analyze_single_image(
                        'E:\trial\my_image.tif',
                        'E:\my output folder',
                        'bright'
                        )

            or, with **cleanup** set to *True* and **hole_fill_area** set to *40*:

            .. code-block:: python

                import PIE
                colony_mask, colony_property_df = \
                    PIE.analyze_single_image(
                        'E:\trial\my_image.tif',
                        'E:\my output folder',
                        'bright',
                        hole_fill_area = 40,
                        cleanup = True
                        )
                        
             .. note::
             
             this example uses Windows path format. For MacOS/Unix Terminal, replace **input_im_path** with the format ~/trial/my_image.tif and **output_path** with the format ~/trial_output_images.

    .. tab:: command-line

        .. code-block:: bash

            pie analyze_single_image INPUT_IM_PATH OUTPUT_PATH IMAGE_TYPE

        or, with options

        .. code-block:: bash

            pie analyze_single_image INPUT_IM_PATH OUTPUT_PATH IMAGE_TYPE -h HOLE_FILL_AREA -c CLEANUP -m MAX_PROPORTION_EXPOSED_EDGE -s SAVE_EXTRA_INFO

        .. note::

            Inputs besides **INPUT_IM_PATH**, **OUTPUT_PATH**, and **IMAGE_TYPE** are optional.

        .. admonition:: Windows cmd Examples
            :class: dropdown, tip

            Analyze an image with bright cells on a dark background, *E:\\trial\\my_image.tif*, and save outputs in a new folder called *E:\\trial_output_images*

            .. code-block:: console

                pie analyze_single_image E:\trial\t01xy0001.tif E:\trial_output_images bright

            or, with **cleanup** set to *True* and **hole_fill_area** set to *40*:

            .. code-block:: console

                pie analyze_single_image E:\trial\t01xy0001.tif E:\trial_output_images bright -h 40 -c True

            .. caution::

                If your filepath has a space in it, you will need to surround the path name with quotation marks, e.g.:

                .. code-block:: console

                    pie analyze_single_image E:\trial\t01xy0001.tif "E:\my output folder" bright

        .. admonition:: MacOS/Unix Terminal Examples
            :class: dropdown, tip

            Analyze an image with bright cells on a dark background, *~/trial/my_image.tif*, and save outputs in a new folder called *~/trial_output_images*

            .. code-block:: console

                pie analyze_single_image ~/trial/t01xy0001.tif ~/trial_output_images bright

            or, with **cleanup** set to *True* and **hole_fill_area** set to *40*:

            .. code-block:: console

                pie analyze_single_image ~/trial/t01xy0001.tif ~/trial_output_images bright -h 40 -c True

            .. caution::

                If your filepath has a space in it, you will need to prepend the space with '\\', e.g.:

                .. code-block:: console

                    pie analyze_single_image ~/trial/t01xy0001.tif ~/my\ output\ folder bright


This function runs PIE, creates output folders within **output_path**, and writes files to:

+ *colony_masks*:
    the colony mask, with each colony labeled in a different number, as a tif file
+ *jpegGRimages*:
    a jpeg of the original image
+ *single_image_colony_centers*:
    a csv file containing the properties (e.g. area) of all the colonies in the image.

If **save_extra_info** is *True* (default), then additional files are saved in the following folders:

+ *boundary_ims*:
    a jpeg of the original image, overlaid with the contours of the colony mask
+ *threshold_plots*:
    plots demonstrating the detection of the threshold based on the log histogram of a background-corrected image, and files with information on curve fits and threshold values for thresholding
+ *colony_center_overlays*:
    a jpeg of the original image, overlaid with the contours of the colony mask and a transparent mask of the cell centers detected after thresholding

