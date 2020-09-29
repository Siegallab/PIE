#!/usr/bin/python

'''
Measures growth across time
'''

import os
import numpy as np
import pandas as pd
import re
import shutil
from PIE import track_colonies, analysis_configuration, colony_filters

class _CompileColonyData(object):
	'''
	Compiles tracked colony data from each position into matrices of
	colony properties across time
	'''
	def __init__(self, analysis_config, colony_data_tracked_df):
		# remove rows where time_tracking_id is None
		self.colony_data_tracked_df = \
			colony_data_tracked_df.dropna(subset = ['time_tracking_id'])
		self.analysis_config = analysis_config
		# identify indices of each unique timepoint and
		# time_tracking_id value in their respective columns
		self._get_index_locations()
		# colony properties for which to NOT make property matrices
		cols_to_exclude = ['timepoint', 'phase_num', 'xy_pos_idx',
			'time_tracking_id', 'main_image_name', 'bb_height', 'bb_width',
			'bb_x_left', 'bb_y_top', 'cross_phase_tracking_id']
		# colony properties for which to make property matrices
		self.cols_to_include = \
			list(set(self.colony_data_tracked_df.columns.to_list()) -
				set(cols_to_exclude))

	def _get_index_locations(self):
		'''
		Identifies indices of every unique value of timepoint and
		time_tracking_id columns in self.colony_data_tracked_df, which
		can then be used to set those values in a numpy array with
		time_tracking_id rows and timepoint columns
		'''
		self.timepoint_list, self.timepoint_indices = \
			np.unique(self.colony_data_tracked_df.timepoint.to_numpy(),
				return_inverse = True)
		self.time_tracking_id_list, self.time_tracking_id_indices = \
			np.unique(self.colony_data_tracked_df.time_tracking_id.to_numpy(),
				return_inverse = True)
		# create a blank matrix of the right dimensions
		self.empty_col_property_mat = \
			np.empty((self.time_tracking_id_list.shape[0],
				self.timepoint_list.shape[0]))
		self.empty_col_property_mat[:] = np.nan

	def _create_property_mat(self, col_property):
		'''
		Reads in dataframe of colony properties with information and
		organizes the data from the column corresponding to col_property
		into a matrix (as a pandas df) with timepoints as column names
		and time_tracking_id as the row names
		'''
		# Indexing is ridiculous in pandas so fill in values in a numpy
		# array, then convert to pandas df and add column and row names
		# Get property values we're interested in as numpy array
		# Unless property vals are strings, need to convert values to
		# float, since only float-type numpy arrays have np.nan values
		col_from_strings = \
			self.colony_data_tracked_df[col_property].dtype == 'O'
		if col_from_strings:
			property_vals = \
				self.colony_data_tracked_df[col_property].to_numpy()
		else:
			property_vals = \
				self.colony_data_tracked_df[col_property]\
					.to_numpy().astype(float)
		# create and populate colony property matrix
		col_property_mat = \
			self.empty_col_property_mat.copy().astype(property_vals.dtype)
		col_property_mat[self.time_tracking_id_indices,
			self.timepoint_indices] = property_vals
		# convert to pandas df
		col_property_df = pd.DataFrame(col_property_mat,
			columns = self.timepoint_list, index = self.time_tracking_id_list)
		# if property values are string, need to replace 'nan' with
		# np.nan
		if col_from_strings:
			col_property_df.replace('nan', np.nan)
		# sort by rows and columns
		col_property_df.sort_index(inplace = True)
		col_property_df.reindex(sorted(col_property_df.columns), axis=1)
		return(col_property_df)

	def _save_property_df(self, col_property, col_property_df):
		'''
		Writes col_property_df to file
		'''
		write_path = self.analysis_config.get_property_mat_path(col_property)
		col_property_df.to_csv(write_path)

	def generate_imaging_info_df(self):
		'''
		Returns a dataframe of imaging info (e.g. xy position, phase_num)
		'''
		imaging_info_cols = \
			['cross_phase_tracking_id', 'time_tracking_id', 'xy_pos_idx', 'phase_num']
		imaging_info_df = \
			self.colony_data_tracked_df[imaging_info_cols].drop_duplicates()
		# imaging_info_df should be indexed by time_tracking_id
		imaging_info_df.set_index('time_tracking_id', inplace = True)
		return(imaging_info_df)

	def generate_property_matrices(self):
		'''
		Generates and saves property matrices for every colony property
		among self.columns_to_include
		Returns matrices for timepoint, area, cX, and cY, which can be
		used in growth rate calculation
		'''
		# set up dictionary that will be returned for growth rate calc
		properties_to_return = ['area', 'time_in_seconds', 'cX', 'cY']
		dict_for_gr = dict.fromkeys(properties_to_return)
		# identify fluorescent properties to return
		fluor_prop_dict = dict()
		# loop through colony properties and create and save a matrix
		# (df) of properties over time for each
		for col_property in self.cols_to_include:
			# create colony property matrix in pandas format
			col_property_df = self._create_property_mat(col_property)
			# save
			self._save_property_df(col_property, col_property_df)
			# if needed, add to dict_for_gr
			if col_property in properties_to_return:
				dict_for_gr[col_property] = col_property_df
			# if fluor channel property, add to fluor_prop_dict
			if '_flprop_' in col_property:
				fluor_prop_dict[col_property] = col_property_df
		return(dict_for_gr['area'], dict_for_gr['time_in_seconds'], dict_for_gr['cX'],
			dict_for_gr['cY'], fluor_prop_dict)

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
				'mindist'] + imaging_info_df.columns.to_list()

	def _convert_times(self, times_in_seconds):
		'''
		Converts times from absolute number of seconds to hours since
		1st image captured
		'''
		### !!! NEEDS UNITTEST
		rel_times_in_seconds = \
			times_in_seconds - self.analysis_config.first_timepoint_time
		rel_times_in_hrs = rel_times_in_seconds.astype(float)/3600
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
		A = np.vstack([x, np.ones(len(x))]).T
		reg_results = np.linalg.lstsq(A, y, rcond=None)
		[slope, intercept] = reg_results[0]
		resid = reg_results[1]
		rsq = 1 - resid / (y.size * y.var())
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

	def _add_fluor_data(self):
		'''
		If there is measured fluorescent data, add the necessary
		timepoints to the growth rate output
		'''
		### !!! NEEDS UNITTEST
		row_indices_to_use = self.final_gr.index
		if self.postphase_analysis_config == None:
			combined_fluor_channel_df = self.analysis_config.fluor_channel_df
		else:
			combined_fluor_channel_df = pd.concat([
					self.analysis_config.fluor_channel_df,
					self.postphase_analysis_config.fluor_channel_df
					], sort = False
				)
		if not combined_fluor_channel_df.empty:
			fl_prop_keys = list(self.fl_prop_mat_df_dict.keys())
			for row_idx, channel in \
				enumerate(combined_fluor_channel_df.\
					fluor_channel_column_name):
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
						combined_fluor_channel_df.at[row_idx, 'fluor_timepoint']
					self.final_gr[fl_prop_name], _ = get_colony_properties(
						col_prop_mat_df,
						timepoint_label,
						index_names = row_indices_to_use,
						gr_df = self.final_gr)

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

def find_column_idx(df, col_vals):
	'''
	Returns indices of col_vals in df.columns
	'''
	# sort columns of df after converting them to same type as col_vals
	df.columns = df.columns.to_numpy().astype(type(col_vals[0]))
	sorter = np.argsort(df.columns)
	col_val_idx = sorter[np.searchsorted(
		df.columns, col_vals, sorter=sorter)]
	return(col_val_idx)

def get_colony_properties(col_property_mat_df, timepoint_label,
	index_names = None, gr_df = None):
	'''
	Returns values from pd df col_property_mat_df at timepoint_label for
	rows with names index_names, as well as timepoints corresponding to those
	values

	If index_names is None, returns properties for all indices

	timepoint_label can be a numpy array of timepoints, a single int,
	'last_tracked' (meaning the last tracked timepoint for each index),
	'first_tracked' (meaning the first tracked timepoint for each
	index),
	'first_gr' (meaning the timepoint corresponding to t0),
	'last_gr' (meaning the timepoint corresponding to tfinal),
	'mean', 'median', 'max', or 'min' (returning specified function on all
	tracked timepoint_label)

	If timepoint_label is last_gr or first_gr, then gr_df must be
	passed
	'''
	### !!! NEEDS UNITTEST
	# if index_names is None, use all of col_property_mat_df in
	# col_property_mat; otherwise, use rows corresponding to specified
	# index_names
	if isinstance(index_names, pd.Index) or \
		isinstance(index_names, list) or isinstance(index_names, np.ndarray):
		col_property_mat_df_subset = col_property_mat_df.loc[index_names]
	elif index_names is None:
		col_property_mat_df_subset = col_property_mat_df
	else:
		raise TypeError("index_names must be a list of index names")
	col_property_mat = col_property_mat_df_subset.to_numpy()
	if isinstance(timepoint_label, str) and \
		timepoint_label in ['mean', 'median', 'min', 'max']:
		# perform the specified function on tracked timepoint_label in each
		# row
		if timepoint_label == 'mean':
			output_vals = np.nanmean(col_property_mat, axis = 1)
		elif timepoint_label == 'median':
			output_vals = np.nanmedian(col_property_mat, axis = 1)
		elif timepoint_label == 'min':
			output_vals = np.nanmin(col_property_mat, axis = 1)
		elif timepoint_label == 'max':
			output_vals = np.nanmax(col_property_mat, axis = 1)
		# tp_to_use_idx should be empty
		tp_to_use_idx = np.empty([col_property_mat.shape[0],1])
		tp_to_use_idx[:] = np.nan
	else:
		# identify timepoint_label to be used for each index
		if isinstance(timepoint_label, str) and \
			timepoint_label in ['first_gr', 'last_gr']:
			# check that gr_df has been passed
			if (gr_df is not None) and isinstance(gr_df, pd.DataFrame) \
				and (not gr_df.empty):
				if len(gr_df.phase_num.unique())>1:
					raise ValueError(
						'gr_df must contain data for no more than one phase_num'
						)
				if gr_df.index.name == 'time_tracking_id':
					gr_df_time_id_indexed = gr_df
				else:
					gr_df_time_id_indexed = gr_df.set_index('time_tracking_id')
				if timepoint_label == 'first_gr':
					# return last tracked timepoint
					tp_to_use = \
						gr_df.loc[
							col_property_mat_df_subset.index, 't0'].to_numpy()
				else:
					tp_to_use = \
						gr_df.loc[
							col_property_mat_df_subset.index, 'tfinal'].to_numpy()
				tp_to_use_idx = \
					find_column_idx(col_property_mat_df_subset,
						[float(tp) for tp in tp_to_use])
			else:
				raise ValueError(
					'If timepoint_label is first_gr or last_gr then a non-'
					'empty pandas dataframe must be passed as gr_df')
		else:
			nan_mat = np.isnan(col_property_mat)
			if isinstance(timepoint_label, str) and \
				timepoint_label == 'first_tracked':
				# return first tracked timepoint
				tp_to_use_idx = \
					np.argmin(nan_mat, axis = 1)
			elif isinstance(timepoint_label, str) and \
				timepoint_label == 'last_tracked':
				# return last tracked timepoint
				tp_to_use_idx = \
					nan_mat.shape[1] - np.argmin(np.fliplr(nan_mat), axis = 1) - 1
			elif isinstance(timepoint_label, int):
				# return specified timepoint
				tp_to_use_idx = np.where(col_property_mat_df_subset.columns == tp_val)[0][0]
			elif isinstance(timepoint_label, np.ndarray) and \
				timepoint_label.shape == (col_property_mat.shape[0],):
				# return timepoint specified for each row
				tp_to_use_idx = \
					find_column_idx(col_property_mat_df_subset.columns,
						timepoint_label.astype(float).astype(str))
			else:
				raise ValueError(
					'timepoint_label may be "first_tracked", "last_tracked", '
					'"first_gr", "last_gr", an integer, or a 1-D numpy '
					'array with as many elements as rows of interest in '
					'col_property_mat_df')
			tp_to_use = col_property_mat_df_subset.columns[tp_to_use_idx]
		output_vals = \
			col_property_mat[np.arange(0,col_property_mat.shape[0]), tp_to_use_idx]
	return(output_vals, tp_to_use)

def _concat_folder_files(dir_to_concat, extension, output_file_path):
	'''
	Reads all files with extension in dir_to_concat and combines them
	into single output_filename
	'''
	file_list = [f for f in os.listdir(dir_to_concat) if f.endswith(extension)]
	with open(output_file_path, 'w') as outfile:
		for fname in file_list:
			fpath = os.path.join(dir_to_concat,fname)
			with open(fpath) as infile:
				outfile.write(infile.read())
			os.remove(fpath)

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
		_concat_folder_files(phase_thresh_folder, 'csv', thresh_output_file)

def measure_growth_rate(analysis_config, postphase_analysis_config,
	time_tracked_phase_pos_data):
	'''
	Compiles data from individual timepoints into a single dataframe of
	tracked colonies, saves property matrices for each tracked colony,
	performs filtering and measures growth rates for all colonies that
	pass filtration
	'''
	colony_data_compiler = \
		_CompileColonyData(analysis_config, time_tracked_phase_pos_data)
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

def run_growth_rate_analysis(analysis_config_file = None,
		analysis_config_obj_df = None,
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
	analysis_config_obj_df = analysis_configuration.check_passed_config(
		analysis_config_obj_df, analysis_config_file)
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

def run_default_growth_rate_analysis(input_path, output_path,
	total_timepoint_num, phase_num = 1,
	hole_fill_area = np.inf, cleanup = False,
	max_proportion_exposed_edge = 0.75,
	im_file_extension = 'tif', minimum_growth_time = 4,
	label_order_list = ['channel', 'timepoint', 'position'],
	total_xy_position_num = 1, first_timepoint = 1,
	timepoint_spacing = 3600, timepoint_label_prefix = 't',
	position_label_prefix = 'xy', main_channel_label = '',
	main_channel_imagetype = 'brightfield', im_format = 'individual',
	extended_display_positions = [1], first_xy_position = 1,
	growth_window_timepoints = 0,
	max_area_pixel_decrease = np.inf, max_area_fold_decrease = 2,
	max_area_fold_increase = 6, min_colony_area = 10,
	max_colony_area = np.inf, min_correlation = 0.9, min_foldX = 0,
	min_neighbor_dist = 5, fluor_channel_scope_labels = '',
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
	# set parent_phase to empty string, since only one phase and no
	# postphase data
	parent_phase = ''
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	analysis_config_file_standin = os.path.join(output_path, 'setup_file.csv')
	parameter_list = [
		'input_path', 'output_path', 'total_timepoint_num',
		'hole_fill_area', 'cleanup', 'max_proportion_exposed_edge',
		'im_file_extension', 'minimum_growth_time', 'label_order_list',
		'total_xy_position_num', 'first_timepoint', 'timepoint_spacing',
		'timepoint_label_prefix', 'position_label_prefix', 'main_channel_label',
		'main_channel_imagetype', 'im_format',
		'extended_display_positions', 'first_xy_position',
		'growth_window_timepoints', 'max_area_pixel_decrease',
		'max_area_fold_decrease', 'max_area_fold_increase', 'min_colony_area',
		'max_colony_area', 'min_correlation', 'min_foldX', 'min_neighbor_dist',
		'fluor_channel_scope_labels', 'fluor_channel_names',
		'fluor_channel_thresholds', 'fluor_channel_timepoints',
		'parent_phase']
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
		'total_xy_position_num', 'extended_display_positions']
	analysis_config_df.PhaseNum[[param in required_global_params for
		param in analysis_config_df.Parameter]] = 'all'
	analysis_config_df.to_csv(analysis_config_file_standin,
		index = False)
#	analysis_config_file_standin.seek(0)
#	test = pd.read_csv(analysis_config_file_standin)
#	print(test)
#	analysis_config_file_standin.seek(0)
	run_growth_rate_analysis(
		analysis_config_file = analysis_config_file_standin,
		repeat_image_analysis_and_tracking =
			repeat_image_analysis_and_tracking)

