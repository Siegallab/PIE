# PIE Readme

- [System requirements](#system-requirements)
- [Installing PIE](#installing-pie)
- [Running PIE](#running-pie)
  * [Running PIE single-image analysis](#running-pie-single-image-analysis)
  * [Running PIE multi-image experiments](#running-pie-multi-image-experiments)
    + [Setting up multi-image experiments](#setting-up-multi-image-experiments)
      - [General setup](#general-setup)
      - [Phases](#phases)
      - [Setup file Examples](#setup-file-examples)
    + [Running the experiment](#running-the-experiment)
    + [Running PIE with batch submission](#running-pie-with-batch-submission)
      - [Analyzing individual imaged xy positions](#analyzing-individual-imaged-xy-positions)
      - [Combining position-wise data and calculating growth rates](#combining-position-wise-data-and-calculating-growth-rates)
  * [Analysis Outputs](#analysis-outputs)
- [Analysis details](#analysis-details)
  * [Growth Rate](#growth-rate)
  * [Lag](#lag)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

---

## System requirements

PIE runs in python, and is compatible with Python 3.8.

The full list of required python packages is listed in the [requirements.txt](https://github.com/Siegallab/PIE/tree/master/requirements.txt) file; these will be automatically installed using the code [below](#installing-pie).

## Installing PIE

After cloning the repository, PIE can easily be installed using [pip](https://pip.pypa.io/en/stable/):

```
git clone git@github.com:Siegallab/PIE.git
cd PIE/
pip install -e .
```

If you see the error 'Permission denied (publickey) error' for git clone, that means you need to add ssh key to your github account for installation. You can do that by following steps in this [link](https://stackoverflow.com/questions/2643502/how-to-solve-permission-denied-publickey-error-when-using-git). If you need more information about how to add the ssh key to your github account via website, you can refer to this [link](https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account).

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

 * `input_im_path`: the path to the image to be analyzed

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

A phase can also be run that takes only fluorescent images for a single timepoint, in which case `parent_phase` should be set to the phase number of the phase containing the brightfield/phase contrast data corresponding to the fluorescent images.

During growth rate analysis, growth rates will be calculated independently for any phase that contains multiple timepoints, but colony identities will be linked across phases.

##### Setup file Examples

Here are some examples of setup files for experiments commonly done in the Siegal Lab:
 * [setup file](https://github.com/Siegallab/PIE/tree/master/sample_PIE_setup_files/gr_phase_setup.csv) for an experiment consisting of only growth rate measurements
 * [setup file](https://github.com/Siegallab/PIE/tree/master/sample_PIE_setup_files/two_phase_setup.csv) for an experiment consisting of two phases of growth rate measurements, the first with two fluorescent channels, the second with a single fluorescent channel

#### Running the experiment

To run a full image analysis experiment using PIE, you can use the `run_growth_rate_analysis` function, which takes a setup file path as input:

```python
import PIE
PIE.run_growth_rate_analysis(analysis_config_file =
    '/local/path/to/PIE/sample_PIE_setup_files/gr_phase_setup.csv')
```
You can also use the `pie` command-line interface:

```console
pie run /local/path/to/PIE/sample_PIE_setup_files/gr_phase_setup.csv
```

Finally, running a single-phase, brightfield-only analysis with all-default inputs, or modifying just a couple of options at a time, can be achieved without a setup file:

```python
import PIE
PIE.run_default_growth_rate_analysis(
    input_path,
    output_path,
    total_timepoint_num)
```

To achieve the same results as using the provided setup file, some options need to be changed when using the default analysis; for example, the sample data can be analyzed as follows:

```python
import PIE
import numpy as np
PIE.run_default_growth_rate_analysis(
    input_path = '/local/path/to/PIE/PIE_test_data/IN/SL_170619_2_GR_small',
    output_path = '/local/path/to/PIE/PIE_test_data/out/SL_170619_2_GR_small',
    total_timepoint_num = 10, total_xy_position_num = 1000,
    timepoint_spacing = 3600, extended_display_positions = [1, 4, 11],
    growth_window_timepoints = 7,
    max_area_pixel_decrease = 500, min_colony_area = 30,
    max_colony_area = np.inf, min_correlation = 0.9, min_neighbor_dist = 100,
    repeat_image_analysis_and_tracking = False)
```

#### Running PIE with batch submission

PIE is set up to easily be run via batch submission. To do this, the colony tracking step for each individual xy position is separated from the steps that compile the data across positions and calculate growth rates.

##### Analyzing individual imaged xy positions

To analyze each imaged position individually (e.g. via batch jobs), you need to pass the integer corresponding to the xy position being analyzed and the path to the setup file. This can be done e.g. for position 11 of the `SL_170619_2_GR_small` sample data either in python:

```python
import PIE
PIE.track_colonies_single_pos(
    11,
    analysis_config_file = 
        '/local/path/to/PIE/sample_PIE_setup_files/gr_phase_setup.csv'
    )
```

or using the command-line interface

```console
pie track_single_position 11 /local/path/to/PIE/sample_PIE_setup_files/gr_phase_setup.csv
```

In addition, the dataframe returned by `PIE.process_setup_file` can be directly passed to `PIE.track_colonies_single_pos` instead of the setup filepath itself, which may be useful when automating image analysis pipelines:

```python
import PIE
config_df = PIE.process_setup_file('/local/path/to/PIE/sample_PIE_setup_files/gr_phase_setup.csv')
PIE.track_colonies_single_pos(
    xy_pos_idx,
    analysis_config_obj_df = config_df
    )
```

##### Combining position-wise data and calculating growth rates

After every imaged xy position has been analyzed, the data can be combined and growth rates can be calculated by simply running `PIE.run_growth_rate_analysis(...)` (or `pie run ...`), as in [*Running the experiment*](#running-the-experiment), with the default `repeat_image_analysis_and_tracking = False` option.

### Analysis Outputs

Outputs of the PIE experiment can be found in the directory provided by the `output_folder` path in the setup file. Each output folder includes one or multiple folders corresponding to phases of the experiment, named **phase_[*phase_name*]**, with all phase-specific data contained within.

Phase-specific output folders contain:

 * **growth_rates.csv** for each phase of the experiment; created only after running `PIE.run_growth_rate_analysis(...)` or `pie run ...`. For all colonies that pass the filtration steps, this file contains:
   + colony growth rates
   + colony lag times
   + if applicable, a cross-section/summary of colony fluorescent data
 * a folder called **positionwise_colony_property_matrices** containing *csv* files for each quantified colony property, tracked across time, for each colony; created only after running `PIE.run_growth_rate_analysis(...)` or `pie run ...`. This is phase-specific data from each column of **colony_properties_combined.csv** (see below) in matrix form, and can be useful for analysis outside the scope of that done by PIE
 * Phase-specific image analysis outputs (see [*Running PIE single-image analysis*](#running-PIE-single-image-analysis), although without a **single_image_colony_centers** folder, as this data is saved in the colony properties file); these are created during the analysis of every individual imaging position.

In addition to the phase-specific folders, the output folder contains:
 * a folder called **positionwise_colony_properties**, which contains the dataframes produced by cross-time and cross-phase colony tracking, saved individually for every position in *parquet* format.
 * **colony_properties_combined.csv**, containing the (unfiltered) properties of every colony identified in every timepoint during analysis, and the colony tracking data, in *csv* format (this is a simple compilation of the data in **positionwise_colony_properties**); created only after running `PIE.run_growth_rate_analysis(...)` or `pie run ...`.
 * **growth_rates_combined.csv**, containing the data from the phase-specific growth rate files but tracked across all phases of the experiment; created only after running `PIE.run_growth_rate_analysis(...)` or `pie run ...`.

## Analysis details

### Growth Rate

### Lag
