PIE Outputs
===========

Outputs of the PIE experiment can be found in the directory provided by the ``output_folder`` path in the setup file. Each output folder includes one or multiple folders corresponding to phases of the experiment, named **phase_[*phase_name*]**, with all phase-specific data contained within.

Phase-specific output folders contain:

+ **growth_rates.csv** for each phase of the experiment; created only after running ``PIE.run_growth_rate_analysis(...)`` or ``pie run ...``. For all colonies that pass the filtration steps, this file contains:
   + colony growth rates
   + colony lag times
   + if applicable, a cross-section/summary of colony fluorescent data
+ a folder called **positionwise_colony_property_matrices** containing *csv* files for each quantified colony property, tracked across time, for each colony; created only after running ``PIE.run_growth_rate_analysis(...)`` or ``pie run ...``. This is phase-specific data from each column of **colony_properties_combined.csv** (see below) in matrix form, and can be useful for performing custom analyses after running PIE
+ phase-specific image analysis outputs (see :doc:`single_im_analysis`, although without a **single_image_colony_centers** folder, as this data is saved in the colony properties file); these are created during the analysis of every individual imaging position
+ **filtered_colonies.csv**, containing a list of colony IDs for colonies that were filtered out at one or more timepoints over the course of the analysis, with the first timepoint a colony is removed due to a particular filter listed in that filter's column. This data can be helpful for advanced users exploring the effects of tuning filtration options in the experimental setup file.

In addition to the phase-specific folders, the output folder contains:

+ **colony_properties_combined.csv**: containing the (unfiltered) properties of every colony identified in every timepoint during analysis, and the colony tracking data, in *csv* format; created only after running ``PIE.run_growth_rate_analysis(...)`` or ``pie run ...``
+ **growth_rates_combined.csv**: containing the data from the phase-specific growth rate files but tracked across all phases of the experiment; created only after running ``PIE.run_growth_rate_analysis(...)`` or ``pie run ...``
+ **setup_file.csv**: a copy of the setup configuration file used for running the experiment
