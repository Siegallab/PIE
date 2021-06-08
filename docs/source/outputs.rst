PIE Outputs
===========

Outputs of the PIE experiment can be found in the directory provided by the ``output_folder`` path in the setup file. Each output folder includes one or multiple folders corresponding to phases of the experiment, named *phase_[phase_num]*, with all phase-specific data contained within (see :ref:`here <phase explanation>` for more information on experimental phases).

Complete time-lapse experiment outputs
--------------------------------------

The final experimental output is created only after running ``PIE.run_timelapse_analysis(...)`` or ``pie run_timelapse_analysis ...``. It contains **setup_file.csv**, a complete version (including all parameter values initially left to the default) of the original setup file used in the experiment.

In addition, the main folder contains general experimental outputs (`Growth Data`_, `Colony properties across time`_, and in some experiments `Movies`_), as well as a folder for each experimental phase containing `phase-specific outputs`_.

Growth data
^^^^^^^^^^^

**growth_rates_combined.csv**: contains the growth data and key summaries of colony properties over time for every colony that passes growth rate filtration. Properties are reported for each phase, and are followed by *_phase_[phase_num]* in the column name.

.. list-table:: Colony growth properties reported in **growth_rates_combined.csv**
    :header-rows: 1
    :stub-columns: 1
    :widths: 1 3

    * - Column
      - Description
    * - cross_phase_tracking_id
      - a unique ID that can be used to identify each colony across experimental phases. If there is only one brightfield/phase-contrast experimental phase, this will be identical to **time_tracking_id**
    * - xy_pos_idx
      - the index of the imaging position the colony is found in
    * - time_tracking_id
      - a unique ID that can be used to identify each colony across time points *within* an experimental phase. If there is only one brightfield/phase-contrast experimental phase, this will be identical to **cross_phase_tracking_id**
    * - t0
      - the first timepoint in the window used to estimate growth rate
    * - tfinal
      - the final timepoint in the window used to estimate growth rate
    * - gr
      - the growth rate of the colony (the natural log of the number of cell divisions per hour)
    * - lag
      - the pre-growth lag time of the colony in hours
    * - intercept
      - the y-intercept of the best-fit line to ln(area) between **t0** and **tfinal** (used for plotting growth rates)
    * - rsq
      - the R\ :superscript:`2` of the best-fit line to ln(area) between **t0** and **tfinal**
    * - foldX
      - the fold increase in colony area over the entire course of the phase
    * - mindist
      - the distance to the closest colony recognized for at least ``minimum_growth_time`` timepoints
    * - cxm
      - the mean across timepoints of the pixel x-position of the weighted colony center
    * - cym
      - the mean across timepoints of the pixel y-position of the weighted colony center

In addition, experiments that include fluorescence data will include fluorescent properties in growth_rates_combined.csv, either summarized or from a single timepoint (depending on the value of the ``fluor_channel_timepoints`` parameter); see `Colony fluorescence properties across time reported in **colony_properties_combined**`_.

Finally, experiments with post-phase fluorescence will include a fluorescence classification column for each channel imaged in a phase's post-phase fluorescence, titled ``fluor_channel_name``_fluorescence_phase_``phase_num``.

Colony properties across time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**colony_properties_combined.csv**: contains the (unfiltered) properties of every colony identified in every timepoint during analysis, and the colony tracking data.

.. list-table:: Colony properties across time reported in **colony_properties_combined**
    :header-rows: 1
    :stub-columns: 1
    :widths: 1 3

    * - Column
      - Description
    * - cross_phase_tracking_id
      - a unique ID that can be used to identify each colony across experimental phases. If there is only one brightfield/phase-contrast experimental phase, this will be identical to **time_tracking_id**
    * - time_tracking_id
      - a unique ID that can be used to identify each colony across time points *within* an experimental phase. If there is only one brightfield/phase-contrast experimental phase, this will be identical to **cross_phase_tracking_id**
    * - main_image_name
      - the name (without file extension) of the image from which colony segmentation was performed to identify a given colony
    * - phase_num
      - the number of the phase in which colony segmentation occurred
    * - xy_pos_idx
      - the index of the imaging position the colony is found in
    * - timepoint
      - the timepoint in which the colony properties are being measured
    * - time_in_seconds
      - the time, in seconds, that elapsed between the start of the experiment and the current **timepoint**. If ``timepoint_spacing`` is provided, **time_in_seconds** is calculated based on the number of seconds to the current **timepoint**, and is identical for a given timepoint all xy_pos_idx values. If ``timepoint_spacing`` is left blank (i.e. is inferred from file modification times), this will be the time between the last modification time of the **main_image_name** image and the first image saved in ``input_path1``
    * - bb_height
      - the height of the bounding box containing the colony
    * - bb_width
      - the width of the bounding box containing the colony
    * - bb_x_left
      - the left-most pixel position of the colony bounding box
    * - bb_y_top
      - the top-most pixel position of the colony bounding box
    * - cX
      - the pixel x-position of the weighted colony center
    * - cY
      - the pixel y-position of the weighted colony center
    * - major_axis_length
      - the length of the major axis of an ellipse fitted to the colony outline. Inaccurate for small colonies (area ≤ 5 pixels)
    * - perimeter
      - the perimeter of the colony
    * - area
      - the area of the colony
    * - label
      - the label of the colony in the ``colony_mask`` image (in the *phase_``phase_num``/colony_mask* directory) corresponding to the current **main_image_name**
    
If fluorescent measurements are made, they will also be listed in **colony_properties_combined.csv** for every colony. Each fluroescent column will be followed by _``fluor_channel_name``; if multiple fluorescent channels are specified, the list of columns below will be reported for each one. All measurements (for both background and colony fluorescence) exclude an approximately 3-pixel-wide strip centered on the colony edge (see the PIE paper for details). Background fluorescence is measured within an area of the colony bounding box (the smallest rectangle containing the identified colony) extended by 5 pixels in all four directions but outside of the colony area itself, or of any other colony within the extended bounding box. Note that measurements made during post-phase fluorescence are recorded as belonging to the phase whose colony segmentation they use.

.. list-table:: Colony fluorescence properties across time reported in **colony_properties_combined**
    :name: Colony fluorescence properties across time reported in **colony_properties_combined**
    :header-rows: 1
    :stub-columns: 1
    :widths: 1 3

    * - Column
      - Description
    * - back_mean_ppix_flprop
      - The mean fluorescence level per pixel of the background surrounding the colony
    * - back_med_ppix_flprop
      - The median fluorescence level per pixel of the background surrounding the colony
    * - back_var_ppix_flprop
      - The variance in fluorescence level per pixel of the background surrounding the colony
    * - col_mean_ppix_flprop
      - The mean fluorescence level per pixel of the colony
    * - col_med_ppix_flprop
      - The median fluorescence level per pixel of the colony
    * - col_upquartile_ppix_flprop
      - The upper quartile of the fluorescence level per pixel of the colony
    * - col_var_ppix_flprop
      - The variance in fluorescence level per pixel of the colony

Movies
^^^^^^

PIE automatically generates a gif-format movie of colony outlines and a colony growth graph for any imaging position index listed in the ``extended_display_positions`` parameter using the ``make_position_movie`` function (see :doc:`movies` for more details) in the *movie* folder, with each movie named with the relevant position index.

Phase-specific outputs
----------------------

Phase-specific output folders contain:

+ **growth_rates.csv**, which contains the same growth rate data as in *growth_rates_combined.csv* but only for the phase in question
+ **first_timepoint.txt**, which contains the time of the first image (this is used during analysis and in the creation of plot movies)
+ a folder called **positionwise_colony_property_matrices** containing *csv* files for each quantified colony property, tracked across time, for each colony; created only after running the ``run_timelapse_analysis`` function. This is phase-specific data from each column of **colony_properties_combined.csv** (see below) in matrix form, and can be useful for performing custom analyses after running PIE.
+ phase-specific image analysis outputs (see :doc:`single_im_analysis`, although without a **single_image_colony_centers** folder, as this data is saved in the colony properties file); these are created during the analysis of every individual imaging position
+ **filtered_colonies.csv**, containing a list of colony IDs for colonies that were filtered out at one or more timepoints over the course of the analysis, with the first timepoint a colony is removed due to a particular filter listed in that filter's column. This data can be helpful for advanced users exploring the effects of tuning filtration options in the experimental setup file.

Temporary outputs
-----------------

In addition to the files described here, PIE creates a number of temporary output files during runtime that are deleted when the experiment is completed. For example, the ``track_colonies_single_pos`` function, which is also run internally during time-lapse experiments, creates a summary file for every position called *xy``xy_pos_idx``.parquet* in a temporary folder called *positionwise_colony_properties*.

