#!/usr/bin/python

'''
Measures growth across time
'''

import os
import numpy as np
import pandas as pd
import re
import shutil
import warnings
from PIE import track_colonies, analysis_configuration, colony_filters, \
	fluor_threshold, colony_prop_compilation, movies

class _DistanceCalculator(object):
	'''
	Calculates distance from each colony with a growth rate to nearest
	recognized colony (regardless of whether it was later filtered out)
	within each phase
	'''
	def __init__(self, cX_df, cY_df, analysis_config):
		self.colony_data_tracked_df = \
			analysis_config.get_colony_data_tracked_df(
				remove_untracked = True)
		if len(np.unique(self.colony_data_tracked_df.phase_num)) > 1:
			raise IndexError(
				'Must use single-phase colony property data')
		self.analysis_config = analysis_config
		# drop NAs from cX_unfilt_df and cY_unfilt_df
		cX_filt_df = cX_df.dropna(how = 'all')
		cY_filt_df = cY_df.dropna(how = 'all')
		# check that cX_unfilt and cY_unfilt have same index in same order
		cX_idx = cX_filt_df.index.to_numpy()
		cY_idx = cY_filt_df.index.to_numpy()
		np.testing.assert_equal(cX_idx, cY_idx)
		if np.array_equal(cX_idx, cY_idx):
			self.centroid_index = cX_idx
		else:
			raise ValueError("Indexes of cX and cY should be the same, " +
				"in the same order")
		# find mean values of cX and cY
		self._find_mean_centers(cX_filt_df, cY_filt_df)

	def _find_mean_centers(self, cX, cY):
		'''
		Identifies mean centers of cX and cY over time
		'''
		self.mean_cX = np.nanmean(cX.to_numpy(), axis = 1)
		self.mean_cY = np.nanmean(cY.to_numpy(), axis = 1)

	def _create_dist_mat(self, position_bool):
		'''
		Creates matrix of distances between every point using
		self.mean_cX and self.mean_cY as point coordinates
		'''
		cX_vals = self.mean_cX[position_bool]
		cY_vals = self.mean_cY[position_bool]
		x_dist = cX_vals - cX_vals[np.newaxis].T
		y_dist = cY_vals - cY_vals[np.newaxis].T
		dist_mat = np.sqrt(x_dist**2 + y_dist**2)
		return(dist_mat)

	def _find_min_distances(self, xy_pos):
		'''
		Finds the distance from every colony to its closest neighbor at
		a single xy imaging position
		'''
		# find colonies corresponding to current position
		current_data = \
			self.colony_data_tracked_df[
				self.colony_data_tracked_df.xy_pos_idx == xy_pos]
		current_colonies = np.unique(current_data.time_tracking_id)
		# identify index positions in self.mean_cX and self.mean_cY that
		# correspond to current_colonies
		position_bool = np.isin(self.centroid_index, current_colonies)
		if np.sum(position_bool) == 0:
			nearest_neighbor_dist = np.array([])
		else:
			# create distance matrix between each colony
			dist_mat = self._create_dist_mat(position_bool)
			# fill diagonal of self.dist_mat with Inf to avoid detection of
			# self as closest match
			np.fill_diagonal(dist_mat, np.inf)
			nearest_neighbor_dist = np.min(dist_mat, axis = 1)
		curr_mindist_df = pd.DataFrame({'mindist': nearest_neighbor_dist},
			index = self.centroid_index[position_bool])
		return(curr_mindist_df)

	def get_mindist_df(self):
		'''
		Creates dataframe of min distance to nearest neighbor for every
		colony
		'''
		mindist_df_list = []
		for xy_pos in self.analysis_config.xy_position_vector:
			curr_mindist_df = self._find_min_distances(xy_pos)
			mindist_df_list.append(curr_mindist_df)
		mindist_df_combined = pd.concat(mindist_df_list, sort = False)
		return(mindist_df_combined)

class _GrowthMeasurer(object):
	'''
	Measures growth rate based on pre-made area and time files
	'''
	# TODO:	A major question about this code concerns the sliding window
	# 		algorithm.
	#		In the old code, any window is allowed as long as it starts
	#		and ends with a measures area (but NaN in the middle are
	#		allowed).
	#		In the current implementation, we are only calculating
	#		growth rate for those windows that don't contain any missing
	#		areas
	def __init__(self, areas, times_in_seconds, cX, cY, fl_prop_mat_df_dict,
			analysis_config, postphase_analysis_config, imaging_info_df):
		self.analysis_config = analysis_config
		self.postphase_analysis_config = postphase_analysis_config
		self.imaging_info_df = imaging_info_df
		self.unfilt_areas = areas
		# times nee
		self.unfilt_times = self._convert_times(times_in_seconds)
		self.unfilt_cX = cX
		self.unfilt_cY = cY
		self.fl_prop_mat_df_dict = fl_prop_mat_df_dict
		self.columns_to_return = \
			['t0', 'tfinal', 'gr', 'lag', 'intercept', 'rsq', 'foldX',
				'mindist', 'cxm', 'cym'] + imaging_info_df.columns.to_list()

	def _convert_times(self, times_in_seconds):
		'''
		Converts times from absolute number of seconds to hours since
		1st image captured
		'''
		### !!! NEEDS UNITTEST
		if len(times_in_seconds.index)>0:
			rel_times_in_seconds = \
				times_in_seconds - self.analysis_config.first_timepoint_time
			rel_times_in_hrs = rel_times_in_seconds.astype(float)/3600
		else:
			rel_times_in_hrs = times_in_seconds
		return(rel_times_in_hrs)

	def _filter_pre_gr(self):
		'''
		Runs pre-growth rate filtering of colonies
		'''
		pre_growth_filter = \
			colony_filters.PreGrowthCombinedFilter(self.analysis_config,
				self.unfilt_areas)
		# filter areas: set any value that needs to be removed in
		# pre-growth filtration to NaN
		self.filt_areas, self.pre_gr_removed_location_df = \
			pre_growth_filter.filter()
		# set any value that was removed from areas in pre-growth
		# filtration to NaN in times, cX, and cY
		self.filt_times = pre_growth_filter.reproduce_filter(self.unfilt_times)
		self.filt_cX = pre_growth_filter.reproduce_filter(self.unfilt_cX)
		self.filt_cY = pre_growth_filter.reproduce_filter(self.unfilt_cY)

	def _identify_regr_window_start_and_ends(self):
		'''
		Identifies all positions in self.log_filt_areas that correspond
		to the start of a consecutive run of
		self.analysis_config.growth_window_timepoints
		Key to this strategy is that self.log_filt_areas has already
		been filtered using
		colony_filters._FilterByGrowthWindowTimepoints (i.e. any
		colonies not part of runs at least
		self.analysis_config.growth_window_timepoints have been removed)
		and is not empty
		'''
		### NEEDS UNITTEST!!!
		# identify position when a consecutive run starts and ends
		# use absolute indices in 1-D array
		non_null_bool = self.log_filt_areas.notnull().to_numpy()
		non_null_bool_diff = np.diff(non_null_bool, prepend = 0, append = 0)
		# count the length of each consecutive run of True at each
		# row and column position in non_null_bool
		non_null_row, non_null_start_col = np.where(non_null_bool_diff == 1)
		_, non_null_stop_col = np.where(non_null_bool_diff == -1)
		subarray_lengths = non_null_stop_col - non_null_start_col
		subarray_flat_start_positions = \
			non_null_start_col + non_null_row*non_null_bool.shape[1]
		flat_cum_run_lengths = colony_filters.cum_sum_since_subarray_start(
			non_null_bool.size, subarray_flat_start_positions, subarray_lengths)
		# reshape back into original shape
		cum_run_lengths = flat_cum_run_lengths.reshape(non_null_bool.shape)
		# set values that are False in non_null_bool to 0, since
		# cumulative sum values returned accumulate in-between subarrays
		cum_run_lengths[np.invert(non_null_bool)] = 0
		# find positions of window ends
		window_rows, window_end_positions = \
			np.where(cum_run_lengths >=
				self.analysis_config.growth_window_timepoints)
		window_start_positions = \
			window_end_positions - \
			self.analysis_config.growth_window_timepoints + 1
		# create dataframe of positions to run linear regression on
		self.positions_for_regression = pd.DataFrame({
			't0': self.log_filt_areas.columns[window_start_positions],
			'tfinal': self.log_filt_areas.columns[window_end_positions],
			'start_col': window_start_positions,
			'range_stop_col': (window_start_positions +
				self.analysis_config.growth_window_timepoints),
			'row': window_rows,
			'intercept': np.nan,
			'gr': np.nan,
			'lag': np.nan,
			'rsq': np.nan,
			'foldX': np.nan
			})
#		# set types of row and column values to int so they can be used
#		# as indices
#		self.positions_for_regression = positions_for_regression.astype(
#			{'start_col': 'int32', 'range_stop_col': 'int32', 'row': 'int32'})

	def _set_up_gr_calc(self):
		'''
		Creates self.log_filt_areas, and empty dataframe that will hold
		growth rate data
		'''
		# Take log of areas for regression on time
		# nb: taking the log of the colony area here is OK because any
		# 0-area colonies have already been filtered out by min_area
		self.log_filt_areas = np.log(self.filt_areas)
		# Get numpy array of log_filt_areas and filt_times to avoid
		# dealing with pandas indexing issues
		self.log_filt_areas_mat = self.log_filt_areas.to_numpy()
		self.filt_times_mat = self.filt_times.to_numpy()
		# set up dataframe of positions to run growth rate calculation
		if self.log_filt_areas.empty:
			# create empty gr dataframe
			self.positions_for_regression = \
				pd.DataFrame(columns = ['t0', 'tfinal', 'start_col',
					'range_stop_col', 'row', 'intercept', 'gr', 'lag',
					'rsq', 'foldX'],
					index = [])
		else:
			self._identify_regr_window_start_and_ends()

	def _run_regression(self, y, x):
		'''
		Regresses y vector on x vector
		'''
		### NEEDS UNITTEST!!!
		# assumes y and x no longer have NaN values
		# assumes x values are not all the same
		A = np.vstack([x, np.ones(len(x))]).T
		reg_results = np.linalg.lstsq(A, y, rcond=None)
		[slope, intercept] = reg_results[0]
		resid = reg_results[1]
		if resid:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				rsq = 1 - resid[0] / (y.size * y.var())
		else:
			rsq = np.nan
		return(slope, intercept, rsq)

	def _calculate_growth_rates(self):
		'''
		Calculates growth rate of all colonies after filtration
		Returns three dataframes: slope_df, intercept_df, and rsq_df
		Each contains a value corresponding to the slope/intercept/rsq
		value of a window starting at that timepoint and containing
		self.analysis_config.growth_window_timepoints consecutive
		timepoints
		'''
		### NEEDS UNITTEST!!!
		# Loop through positions specified in
		# self.positions_for_regression, running a linear regression on
		# log(area) over time for each one and recording the results
		for regr_pos_info in self.positions_for_regression.itertuples():
			idx = regr_pos_info.Index
			row = regr_pos_info.row
			start = regr_pos_info.start_col
			stop = regr_pos_info.range_stop_col
			log_areas = self.log_filt_areas_mat[row, start:stop]
			times = self.filt_times_mat[row, start:stop]
			if len(np.unique(times)) < len(times):
				warnings.warn(
					"Multiple identical time values: "+np.array2string(times),
					UserWarning
					)
			slope, intercept, rsq = self._run_regression(log_areas, times)
			self.positions_for_regression.at[idx, 'gr'] = slope
			self.positions_for_regression.at[idx, 'intercept'] = intercept
			self.positions_for_regression.at[idx, 'rsq'] = rsq
			self.positions_for_regression.at[idx, 'foldX'] = \
				np.exp(np.max(log_areas) - np.min(log_areas))

	def _calculate_lags(self):
		'''
		Calculates lag following Naomi Ziv's method:
		Lag duration is estimated as the intersection of regression line
		with a horizontal line determined by the area of the microcolony
		at the first time point
		Lag duration is not calculated for colonies that are not tracked
		in the first time point
		(Ziv et al 2013)
		NB: Unlike Ziv et al 2013, here lag is calculated relative to
		the time the first image is collected, not the time the first
		image of the current field is collected
		'''
		if not self.log_filt_areas.empty:
			initial_log_areas = \
				self.log_filt_areas_mat[self.positions_for_regression.row, 0]
			self.positions_for_regression.lag = \
				(initial_log_areas -
					self.positions_for_regression.intercept) / \
					self.positions_for_regression.gr

	def _apply_mindist(self, df):
		'''
		Adds vals from self.mindist_df to df so they can be used in
		post-growth filtration
		'''
		distance_calculator = \
			_DistanceCalculator(self.filt_cX, self.filt_cY,
				self.analysis_config)
		mindist_df = distance_calculator.get_mindist_df()
		df_with_dist = df.join(mindist_df)
		return(df_with_dist)

	def _add_center_data(self, df):
		'''
		Adds data for mean center position of each colony
		'''
		### !!! NEEDS UNITTEST
		row_indices_to_use = df.index
		df['cxm'], _ = \
			colony_prop_compilation.get_colony_properties(
				self.filt_cX,
				'mean',
				index_names = row_indices_to_use)
		df['cym'], _ = \
			colony_prop_compilation.get_colony_properties(
				self.filt_cY,
				'mean',
				index_names = row_indices_to_use)
		return(df)

	def _filter_post_gr(self):
		'''
		Runs post-growth rate filtering of colonies
		'''
		### NEEDS UNITTEST!!!
		# create a copy of self.positions_for_regression, and set its
		# index to colony ID to allow easy recording of filtered-out
		# colonies
		prefilt_growth_rate_data = self.positions_for_regression.copy()
		prefilt_growth_rate_data.index = \
			self.log_filt_areas.index[prefilt_growth_rate_data.row.to_list()]
		# add vals for distance to closest neighboring colony to the df
		prefilt_growth_rate_data = self._apply_mindist(prefilt_growth_rate_data)
		# add mean colony center positions to the df
		prefilt_growth_rate_data = \
			self._add_center_data(prefilt_growth_rate_data)
		# create filtration object
		post_growth_filter = \
			colony_filters.PostGrowthCombinedFilter(self.analysis_config,
				prefilt_growth_rate_data)
		# filter growth: set any value that needs to be removed in
		# post-growth filtration to NaN, and remove all-NaN rows
		self.filt_growth_rates, self.post_gr_removed_location_df = \
			post_growth_filter.filter()

	def _select_colony_gr(self):
		'''
		Among filtered growth rates, selects the max calculated growth
		rate for each colony
		'''
		### NEEDS UNITTEST!!!
		# re-index self.filt_growth_rates, renaming current index to
		# 'time_tracking_id', so that indices can be used below to
		# select maximal growth rates
		self.filt_growth_rates.rename_axis('time_tracking_id',
			inplace = True )
		self.filt_growth_rates.reset_index(inplace = True)
		# identify locations of max growth rate for each colony id
		max_gr_idx = \
			self.filt_growth_rates.groupby('time_tracking_id')['gr'].idxmax()
		filtered_gr = self.filt_growth_rates.loc[max_gr_idx]
		# re-index with time_tracking_id
		filtered_gr.set_index('time_tracking_id', inplace = True)
		# join filtered gr and imaging info gr by their indices
		combined_filtered_gr = filtered_gr.join(self.imaging_info_df)
		self.final_gr = combined_filtered_gr[self.columns_to_return]

	def _classify_postphase_fluor(self):
		'''
		Create a bool column determining whether postphase 
		fluorescence is present based on fitting 
		bimodal distribution to fluorescence data
		'''
		# set property to use for classification of post-phase 
		# fluorescent data
		class_prop = 'col_mean_ppix_flprop'
		for channel in \
			self.postphase_analysis_config.fluor_channel_df.\
				fluor_channel_column_name:
			channel_fluor_data_name=class_prop+'_'+channel
			channel_bool_name = channel+'_fluorescence'
			fluor_logistic_finder = \
				fluor_threshold.FluorLogisticFitter(
					self.final_gr[channel_fluor_data_name].to_numpy()
					)
			fluor_classification_file_path = os.path.join(
				self.analysis_config.phase_output_path,
				channel+'_classification_output_file.csv'
				)
			fluor_classification_plot_path = os.path.join(
				self.analysis_config.phase_output_path,
				channel+'_classification_plot.pdf'
				)
			self.final_gr[channel_bool_name] = \
				fluor_logistic_finder.classify_data(
					fluor_classification_file_path,
					channel,
					fluor_classification_plot_path,
					6.5,
					4.5
					)

	def _add_fluor_data(self):
		'''
		If there is measured fluorescent data, add the necessary
		timepoints to the growth rate output
		'''
		### !!! NEEDS UNITTEST
		row_indices_to_use = self.final_gr.index
		if self.postphase_analysis_config is None:
			combined_fluor_channel_df = self.analysis_config.fluor_channel_df.copy()
		else:
			combined_fluor_channel_df = pd.concat([
					self.analysis_config.fluor_channel_df,
					self.postphase_analysis_config.fluor_channel_df
					], sort = False
				)
		if not combined_fluor_channel_df.empty:
			# set index to fluorescent channel name
			combined_fluor_channel_df.set_index(
				'fluor_channel_column_name',
				inplace = True
				)
			fl_prop_keys = list(self.fl_prop_mat_df_dict.keys())
			for channel in combined_fluor_channel_df.index:
				# identify property names in this channel
				prop_substring = '_flprop_' + channel
				prop_substring_re = re.compile('.*'+prop_substring+'$')
				current_channel_prop_names = \
					list(filter(prop_substring_re.match, fl_prop_keys))
				# loop through property names for current channel, read
				# them in, identify data at timepoints, add as column to
				# self.final_gr
				for fl_prop_name in current_channel_prop_names:
					col_prop_mat_df = self.fl_prop_mat_df_dict[fl_prop_name]
					timepoint_label = \
						combined_fluor_channel_df.at[channel, 'fluor_timepoint']
					self.final_gr[fl_prop_name], _ = \
						colony_prop_compilation.get_colony_properties(
							col_prop_mat_df,
							timepoint_label,
							index_names = row_indices_to_use,
							gr_df = self.final_gr)
		# classify fluor data
		if self.postphase_analysis_config is not None:
			self._classify_postphase_fluor()

	def find_growth_rates(self):
		'''
		Finds the growth rate of colonies in self.unfilt_areas, with
		some filtration steps
		'''
		# run filter to remove problematic colonies or timepoints before
		# growth rate analysis
		self._filter_pre_gr()
		# set up data structures needed for growth rate estimation
		self._set_up_gr_calc()
		# calculate growth rates at all possible sets of positions
		self._calculate_growth_rates()
		# calculate lags for all colonies where colony area is measured
		# at the first timepoint
		self._calculate_lags()
		# run filter to remove growth rates with problematic properties
		# (e.g. poor correaltion, too little growth)
		self._filter_post_gr()
		# select the highest growth rate for each colony to report
		self._select_colony_gr()
		# add fluorescent data if it exists
		self._add_fluor_data()
		# save results
		self.final_gr.to_csv(self.analysis_config.phase_gr_write_path)
		return(self.final_gr)

	def get_filtered_colonies(self):
		'''
		Concatenates and returns dataframe containing which colonies
		were filtered out at which timepoints during growth rate
		analysis
		'''
		combined_filter_df = \
			self.pre_gr_removed_location_df.join(
				self.post_gr_removed_location_df, how = 'outer'
				)
		return(combined_filter_df)

class _PhaseGRCombiner(object):
	'''
	Class that combines gr data across phases
	'''
	def __init__(self):
		self.time_tracked_col_prop_list = []
		self.gr_df_list = []

	def add_phase_data(self, gr_df):
		'''
		Add gr and tracked colony property data for a phase to the phase
		combiner
		'''
		# reset indices so we can concatenate and merge by columns later
		gr_df.reset_index(inplace = True)
		self.gr_df_list.append(gr_df)

	def track_gr_data(self, tracked_col_props):
		'''
		Combine growth rate data across phases based on colony tracking
		tracked_col_props should have tracking info across both time
		('time_tracking_id') and phase ('cross_phase_tracking_id')
		'''
		# combine growth rates data across phases - resetting index is
		# important since it is not enforced to be unique across phases
		gr_df_comb = pd.concat(self.gr_df_list, sort = False).reset_index(
			drop = True)
#		phase_match_key_df = \
#			track_colonies.generate_match_key_df(tracked_col_props)
		# re-orient df so that each row is a single colony tracked
		# across phases, with columns from gr_df reported for each phase
		gr_df_comb_squat = \
			gr_df_comb.pivot(index = ['cross_phase_tracking_id', 'xy_pos_idx'],
				columns = 'phase_num')
		# combine multi-indexing in columns to make data R-readable
		gr_df_comb_squat.columns = [f'{i}_phase_{j}' for i,j in gr_df_comb_squat.columns]
		return(gr_df_comb_squat)

#def _concat_folder_files(dir_to_concat, extension, output_file_path):
#	'''
#	Reads all files with extension in dir_to_concat and combines them
#	into single output_file_path
#
#	Will only add lines to output_file_path (not replace it), and will
#	only do so if they're not already there (this prevents issue of 
#	when output_file_path has already been created and input files
#	destroyed, but code is run again)
#	'''
#	file_list = [f for f in os.listdir(dir_to_concat) if f.endswith(extension)]
#	output_path_base = os.path.basename(output_file_path)
#	if output_path_base in file_list:
#		file_list.remove(output_path_base)
#	# code based on https://kaijento.github.io/2017/03/22/python-add-line-to-file-if-not-already-there/
#	with open(output_file_path, 'a+') as outfile:
#		outfile_text = outfile.read()
#		for fname in file_list:
#			fpath = os.path.join(dir_to_concat,fname)
#			if os.path.isfile(fpath):
#				with open(fpath, 'r') as infile:
#					infile_text = infile.read()
#					if not re.search(r'(?m)^'+infile_text, outfile_text):
#						outfile.write(infile_text)
#				os.remove(fpath)
def _concat_csv_files(dir_to_concat, output_file_path):
	'''
	Reads all csv files in dir_to_concat as dataframes and combines 
	them into single df in output_file_path

	Will only add lines to output_file_path (not replace it), and will
	only do so if they're not already there (this prevents issue of 
	when output_file_path has already been created and input files
	destroyed, but code is run again)
	'''
	if os.path.isdir(dir_to_concat):
		file_list = [f for f in os.listdir(dir_to_concat) if f.endswith('csv')]
		df_list = []
		if os.path.isfile(output_file_path):
			output_path_base = os.path.basename(output_file_path)
			file_list.remove(output_path_base)
			output_df_part = pd.read_csv(output_file_path)
			df_list.append(output_df_part)
		for fname in file_list:
			fpath = os.path.join(dir_to_concat,fname)
			input_df = pd.read_csv(fpath, index_col = 0)
			df_list.append(input_df)	
			os.remove(fpath)
		if df_list:
			output_df = pd.concat(df_list).drop_duplicates()
			output_df.to_csv(output_file_path)
	else:
		warnings.warn(
					f"No such directory: {dir_to_concat}\n"
					"You are likely seeing this warning because no image files "
					"were found or colonies were detected",
					UserWarning
					)

def _post_analysis_file_process(output_path, phase_list,
		col_properties_output_folder):
	'''
	Runs file processing post-analysis:
	-	Deletes field-specific tracking data folder
	-	Compiles threshold output csv files into single file and deletes them
	'''
	# delete field-specific tracking data
	shutil.rmtree(col_properties_output_folder)
	# compile threshold output files
	for phase_num in phase_list:
		phase_thresh_folder = \
			os.path.join(
				output_path, 'phase_'+str(phase_num), 'threshold_plots'
				)
		thresh_output_file = \
			os.path.join(phase_thresh_folder, 'threshold_info_comb.csv')
		_concat_csv_files(phase_thresh_folder, thresh_output_file)

def measure_growth_rate(analysis_config, postphase_analysis_config,
	time_tracked_phase_pos_data):
	'''
	Compiles data from individual timepoints into a single dataframe of
	tracked colonies, saves property matrices for each tracked colony,
	performs filtering and measures growth rates for all colonies that
	pass filtration
	'''
	colony_data_compiler = \
		colony_prop_compilation.CompileColonyData(
			analysis_config, time_tracked_phase_pos_data
			)
	area_mat_df, timepoint_mat_df, cX_mat_df, cY_mat_df, fl_prop_mat_df_dict = \
		colony_data_compiler.generate_property_matrices()
	imaging_info_df = colony_data_compiler.generate_imaging_info_df()
	growth_measurer = \
		_GrowthMeasurer(area_mat_df, timepoint_mat_df, cX_mat_df, cY_mat_df,
			fl_prop_mat_df_dict, analysis_config, postphase_analysis_config,
			imaging_info_df)
	gr_df = growth_measurer.find_growth_rates()
	filtered_colony_df = growth_measurer.get_filtered_colonies()
	return(gr_df, filtered_colony_df)

def run_timelapse_analysis(analysis_config_file,
		repeat_image_analysis_and_tracking = False):
	'''
	Runs growth rate analysis from scratch (including image analysis
	steps) based on analysis_config_file
	If repeat_image_analysis_and_tracking is False, and phase colony
	properties file already exists, skip image analysis and tracking
	and go straight to growth rate assay
	'''
	# check that only analysis_config_obj_df or
	# analysis_config_file is passed, and get analysis_config_obj_df
	analysis_config_obj_df = analysis_configuration.process_setup_file(
		analysis_config_file
		)
	phase_gr_combiner = _PhaseGRCombiner()
	# read in colony properties csv if it exists and
	# repeat_image_analysis_and_tracking is False
	temp_analysis_config = analysis_config_obj_df.iloc[0]['analysis_config']
	tracked_prop_file = \
		temp_analysis_config.combined_tracked_properties_write_path
	if not repeat_image_analysis_and_tracking and \
		os.path.isfile(tracked_prop_file):
		time_tracked_col_props_comb = pd.read_csv(tracked_prop_file)
	else:
		# track colonies through time and phase at each xy position and
		# combine results
		# xy positions and track_colonies.track_colonies_single_pos output
		# should be the same for every phase
		time_tracked_col_props_pos_list = []
		for xy_pos_idx in temp_analysis_config.xy_position_vector:
			temp_analysis_config.set_xy_position(xy_pos_idx)
			if repeat_image_analysis_and_tracking or not \
				os.path.exists(temp_analysis_config.tracked_properties_write_path):
				time_tracked_pos_col_props = \
					track_colonies.track_colonies_single_pos(xy_pos_idx,
						analysis_config_obj_df = analysis_config_obj_df)
			else:
				time_tracked_pos_col_props = \
					temp_analysis_config.get_position_colony_data_tracked_df()
			time_tracked_col_props_pos_list.append(time_tracked_pos_col_props)
		# combine time-tracked colony properties for each position into a
		# single df
		time_tracked_col_props_comb = \
			pd.concat(time_tracked_col_props_pos_list, sort = False)
		# write output file
		time_tracked_col_props_comb.to_csv(
			tracked_prop_file, index = False)
	# measure growth rates within each phase
	phase_list = analysis_config_obj_df.index
	for phase_num in phase_list:
		analysis_config = \
			analysis_config_obj_df.at[phase_num, 'analysis_config']
		postphase_analysis_config = \
			analysis_config_obj_df.at[phase_num, 'postphase_analysis_config']
		time_tracked_col_props_phase = \
			time_tracked_col_props_comb[
				time_tracked_col_props_comb.phase_num == phase_num]
		gr_df, filtered_colony_df = measure_growth_rate(analysis_config, postphase_analysis_config,
			time_tracked_col_props_phase)
		phase_gr_combiner.add_phase_data(gr_df)
		# write filtered colony output
		filtered_colony_df.to_csv(analysis_config.filtered_colony_file)
	# combine growth rates across phases
	gr_df_combined = \
		phase_gr_combiner.track_gr_data(time_tracked_col_props_comb)
	gr_df_combined.to_csv(temp_analysis_config.combined_gr_write_path)
	# post-analysis folder and file cleanup
	_post_analysis_file_process(
		temp_analysis_config.output_path,
		phase_list,
		temp_analysis_config.col_properties_output_folder)
	# add movies for extended_display_positions
	for pos in temp_analysis_config.extended_display_positions:
		movies.make_position_movie(
			pos,
			analysis_config_obj_df = analysis_config_obj_df,
			colony_subset = 'growing'
			)

def run_default_growth_rate_analysis(input_path, output_path,
	max_timepoint_num, phase_num = 1,
	hole_fill_area = np.inf, cleanup = False,
	max_proportion_exposed_edge = 0.75,
	cell_intensity_num = 1,
	perform_registration = True,
	im_file_extension = 'tif', minimum_growth_time = 4,
	label_order_list = ['channel', 'timepoint', 'position'],
	max_xy_position_num = 1, first_timepoint = 1,
	timepoint_spacing = 3600, timepoint_label_prefix = 't',
	position_label_prefix = 'xy', main_channel_label = '',
	main_channel_imagetype = 'bright', im_format = 'individual',
	extended_display_positions = [1], first_xy_position = 1,
	growth_window_timepoints = 0,
	max_area_pixel_decrease = np.inf, max_area_fold_decrease = 2,
	max_area_fold_increase = 6, min_colony_area = 10,
	max_colony_area = np.inf, min_correlation = 0.9, min_foldX = 0,
	min_neighbor_dist = 5, max_colony_num = 1000, fluor_channel_scope_labels = '',
	fluor_channel_names = '', fluor_channel_thresholds = '',
	fluor_channel_timepoints = '',
	repeat_image_analysis_and_tracking = True):
	'''
	Runs growth rate analysis on a single phase of brightfield-only
	imaging from scratch (including image analysis steps) based on
	options, not analysis_config_file
	User needs to provide input and output paths, as well as
	the number of total timepoints
	If repeat_image_analysis_and_tracking is False, and phase colony
	properties file already exists, skip image analysis and tracking
	and go straight to growth rate assay
	'''
	# set linked_phase to empty string, since only one phase and no
	# postphase data
	linked_phase = ''
	try:
		os.makedirs(output_path)
	except:
		pass
	analysis_config_file_standin = os.path.join(output_path, 'setup_file.csv')
	parameter_list = [
		'input_path', 'output_path', 'max_timepoint_num',
		'hole_fill_area', 'cleanup', 'max_proportion_exposed_edge',
		'cell_intensity_num', 'perform_registration',
		'im_file_extension', 'minimum_growth_time', 'label_order_list',
		'max_xy_position_num', 'first_timepoint', 'timepoint_spacing',
		'timepoint_label_prefix', 'position_label_prefix', 'main_channel_label',
		'main_channel_imagetype', 'im_format',
		'extended_display_positions', 'first_xy_position',
		'growth_window_timepoints', 'max_area_pixel_decrease',
		'max_area_fold_decrease', 'max_area_fold_increase', 'min_colony_area',
		'max_colony_area', 'min_correlation', 'min_foldX', 'min_neighbor_dist',
		'max_colony_num', 'fluor_channel_scope_labels', 'fluor_channel_names',
		'fluor_channel_thresholds', 'fluor_channel_timepoints',
		'linked_phase']
	param_val_list = []
	for param in parameter_list:
		param_val = eval(param)
		if isinstance(param_val, list):
			param_val_str = \
				';'.join([str(param_val_memb) for param_val_memb in param_val])
		else:
			param_val_str = str(param_val)
		param_val_list.append(param_val_str)
	analysis_config_df = pd.DataFrame({
		'Parameter': parameter_list,
		'Value': param_val_list,
		'PhaseNum': str(phase_num)})
	# need to change PhaseNum in required global parameters to 'all'
	required_global_params = [
		'output_path', 'im_format', 'first_xy_position',
		'max_xy_position_num', 'extended_display_positions']
	analysis_config_df.PhaseNum[[param in required_global_params for
		param in analysis_config_df.Parameter]] = 'all'
	analysis_config_df.to_csv(analysis_config_file_standin,
		index = False)
#	analysis_config_file_standin.seek(0)
#	test = pd.read_csv(analysis_config_file_standin)
#	print(test)
#	analysis_config_file_standin.seek(0)
	run_timelapse_analysis(
		analysis_config_file_standin,
		repeat_image_analysis_and_tracking =
			repeat_image_analysis_and_tracking)

