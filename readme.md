# PIE Readme
---

## System requirements

PIE runs in python, and is compatible with Python 3.8.

The full list of required python packages is listed in the [requirements.txt](https://github.com/Siegallab/PIE/tree/master/requirements.txt) file.

## Installing PIE

After cloning the repository, PIE can easily be installed using [pip](https://pip.pypa.io/en/stable/):

```
git clone git@github.com:Siegallab/PIE.git
cd PIE/
pip install -e .
```

## Calling PIE within other python code

For now, there's no way to actually install PIE properly (currently working on this). To import PIE, you need to run the following in python at runtime (changing `/local/path/to/PIE` to whatever the path is to the PIE repository on your computer; it should end with 'PIE'):
   ```python
   import sys
   sys.path.append('/local/path/to/PIE')
   import PIE
   ```

## Running PIE

### Running PIE single-image analysis

To analyze a single image using PIE, you can use the `analyze_single_image` function, which takes an image filename (and some analysis parameters) as inputs:

 ```python
 import PIE
 colony_mask, colony_property_df = \
     PIE.analyze_single_image(input_im_path, output_path, image_type,
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

 * `max_proportion_exposed_edge`: maximum proportion (0-1) of the perimeter of a PIE-detected gradient object ('PIE piece') that may be non-adjacent to another PIE piece to avoid being removed during 'cleanup' steps; *0-1*, only used if `cleanup` is *True*. 0.75 works well for 10x yeast images

 * `save_extra_info`: whether to write additional files described above; *True* or *False*

 The function returns:

 * `colony_mask`: a numpy boolean matrix with True at positions corresponding to pixels in the original image where a colony was detected

 * `colony_property_df`: a pandas dataframe containing the properties of every colony in the image (same as the ones saved to **single_image_colony_centers**, but also containing a list of all the pixels in which each colony was detected).

### Running PIE multi-image experiments

#### Setting up multi-image experiments

Complete PIE experiments require a single *csv*-format setup file to set up. Some templates for experiment types commonly run in the Siegal lab can be found in [sample_PIE_setup_files](https://github.com/Siegallab/PIE/tree/master/sample_PIE_setup_files).

##### General setup

To set up a file for an experiment, modify an existing setup file (or make your own from scratch, although this is not recommended!) More detailed explanations of all the parameters will be added later, but for now, you can just read the *Explanation* column for each parameter in the setup files provided.

##### Phases

Each experiment may consist of one or more phases; the list of phases in the experiment must be provided in the experimental setup file. A single phase consists of a single, continuously labeled bout of imaging. Colony outlines are always calculated based on a "main channel", which should consist of either brightfield or phase contrast images; the colonies identified in the main channel will then be overlaid on any fluorescent images in the phase to calculate fluorescence levels.

A phase can also be run that takes only fluorescent images for a single timepoint, in which case `main_channel_label` should be set to `phase_#`, where `#` indicates the number of the parent phase.

During growth rate analysis, growth rates will be calculated independently for any phase that contains multiple timepoints, but colony identities will be linked across phases.

##### Setup file Examples

Here are some examples of setup files for experiments commonly done in the Siegal Lab:
 * [setup file for an experiment consisting of only growth rate measurements](https://github.com/Siegallab/PIE/tree/master/sample_PIE_setup_files/gr_phase_setup.csv)

#### Running the experiment

To run a full image analysis experiment using PIE, you can use the `run_growth_rate_analysis` function, which takes a setup file path as input:

 ```python
 import PIE
 PIE.run_growth_rate_analysis('/local/path/to/PIE/sample_PIE_setup_files/gr_phase_setup.csv')
 ```

#### Analysis Outputs

Outputs of the PIE experiment can be found in the directory provided by the `output_folder` path in the setup file.

Each output folder includes one or multiple folders corresponding to phases of the experiment, and named **phase_[*phase_name*]**. This folder contains:

 * **growth_rates.csv**, containing all the colony growth rates for this experiment that pass the filtration steps
 * **col_props_with_tracking_pos.csv**, containing the (unfiltered) properties of every colony identified in every timepoint during analysis, and the colony tracking data
 * Phase-specific image analysis outputs (see [*Running PIE single-image analysis*](#Running-PIE-single-image-analysis), although without a **single_image_colony_centers** folder, as this data is saved in the colony properties file)

In addition, the file containing the growth rates of all colonies tracked across all phases can be found directly in the output folder (**UNDER CONSTRUCTION**)

## Analysis details

### Growth Rate

### Lag
