# PIE Readme
---

## System requirements

PIE runs in python, and is compatible with Python 2.7+

The full list of required python packages is listed in the [requirements.txt](https://github.com/Siegallab/PIE/tree/master/PIE_python/requirements.txt) file; for now, to run PIE, you have to manually install all these packages (however, most can easily be installed using [pip](https://pip.pypa.io/en/stable/))

## Installing PIE

To install PIE, download the directory using the button near the top right of this screen. Then install all the python packages listed in the [requirements.txt](https://github.com/Siegallab/PIE/tree/master/PIE_python/requirements.txt) file.

## Running PIE

### Calling PIE within other python code

To call PIE within other python code, you may:
1. Run PIE with an image filename as the input
   ```python
   from PIE.colony_edge_detection import read_and_run_analysis
   colony_mask, colony_property_df = \
       read_and_run_analysis(input_im_path, output_path, image_type,
           hole_fill_area, cleanup, max_proportion_exposed_edge,
           save_extra_info)
   ```
   This function runs PIE, creates output folders within `output_path`, and writes files containing the binary colony mask as an 8-bit tif file (**colony_masks** folder), a jpeg of the original image (**jpegGRimages** folder), and a csv file containing the properties (e.g. area) of all the colonies in the image (**single_image_colony_centers** folder). If save_extra_info is *True*, then additional files are saved in the following folders:

   * **boundary_ims**: a jpeg of the original image, overlaid with the contours of the colony mask

   * **threshold_plots**: plots demonstrating the detection of the threshold based on the log histogram of a background-corrected image

   * **colony_center_overlays**: a jpeg of the original image, overlaid with the contours of the colony mask and a transparent mask of the cell centers detected after thresholding

   The inputs into this functions are as follows:

   * `input_im_path`: the directory in which the image to be analyzed is saved

   * `output_path`: the directory in which image analysis folders should be created

   * `image_type`: The type of micrscopy used to generate images; *'brightfield'* or *'phasecontrast'*

   * `hole_fill_area`: the area (in pixels) of the largest size hole to fill in colony masks after image analysis. For low-res yeast imaging, we recommend setting this value to numpy.inf (i.e. all the holes in the colonies get filled) *0-numpy.inf*

   * `cleanup`: whether or not to perform 'cleanup' of spurious pieces of background attached to colonies; *True* or *False*. (we recommend trying PIE both with and without cleanup on a set of sample images; you can see the Li, Plavskin *et al.* paper for details)

   * `max_proportion_exposed_edge`: minimum proportion (0-1) of the perimeter of a PIE-detected gradient object ('PIE piece') that must be adjacent to another PIE piece to avoid being removed during 'cleanup' steps; *0-1*, only used if `cleanup` is *True*. 0.25 works well for 10x yeast images

   * `save_extra_info`: whether to write additional files described above; *True* or *False*

   The function returns:

   * `colony_mask`: a numpy boolean matrix with True at positions corresponding to pixels in the original image where a colony was detected

   * `colony_property_df`: a pandas dataframe containing the properties of every colony in the image (same as the ones saved to **single_image_colony_centers**, but also containing a list of all the pixels in which each colony was detected).
