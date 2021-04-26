Running PIE single-image analysis
=================================

To analyze a single image using PIE, you can use the ``analyze_single_image`` function in python, which takes an image filename (and some analysis parameters) as inputs: ::

    import PIE

    colony_mask, colony_property_df = \
        PIE.analyze_single_image(
            input_im_path,
            output_path,
            image_type,
            hole_fill_area,
            cleanup,
            max_proportion_exposed_edge,
            save_extra_info
            )

This function runs PIE, creates output folders within ``output_path``, and writes files containing the binary colony mask as an 8-bit tif file (**colony_masks** folder), a jpeg of the original image (**jpegGRimages** folder), and a csv file containing the properties (e.g. area) of all the colonies in the image (**single_image_colony_centers** folder). If ``save_extra_info`` is ``True``, then additional files are saved in the following folders:

+ **boundary_ims**: a jpeg of the original image, overlaid with the contours of the colony mask
+ **threshold_plots**: plots demonstrating the detection of the threshold based on the log histogram of a background-corrected image
+ **colony_center_overlays**: a jpeg of the original image, overlaid with the contours of the colony mask and a transparent mask of the cell centers detected after thresholding

 The inputs into this functions are as follows:

+ ``input_im_path``: the path to the image to be analyzed
+ ``output_path``: the directory in which image analysis folders should be created
+ ``image_type``: The type of micrscopy used to generate images; *'brightfield'* or *'phasecontrast'*
+ ``hole_fill_area``: the area (in pixels) of the largest size hole to fill in colony masks after image analysis. For low-res yeast imaging, we recommend setting this value to numpy.inf (i.e. all the holes in the colonies get filled) *0-numpy.inf*
+ ``cleanup``: whether or not to perform 'cleanup' of spurious pieces of background attached to colonies; *True* or *False*. (we recommend trying PIE both with and without cleanup on a set of sample images; you can see the Li, Plavskin *et al.* paper for details)
+ ``max_proportion_exposed_edge``: maximum proportion (0-1) of the perimeter of a PIE-detected gradient object ('PIE piece') that may be non-adjacent to another PIE piece to avoid being removed during 'cleanup' steps; *0-1*, only used if ``cleanup`` is *True*. 0.75 works well for 10x yeast images
+ ``save_extra_info``: whether to write additional files described above; *True* or *False*

The function returns:

+ ``colony_mask``: a numpy boolean matrix with True at positions corresponding to pixels in the original image where a colony was detected
+ ``colony_property_df``: a pandas dataframe containing the properties of every colony in the image (same as the ones saved to **single_image_colony_centers**, but also containing a list of all the pixels in which each colony was detected).