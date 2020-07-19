#!/usr/bin/python

'''
Measures growth across time
'''

import os
import numpy as np
import pandas as pd
from PIE import track_colonies, analysis_configuration, colony_filters

class _CompileColonyData(object):
	'''
	Compiles tracked colony data from each position into matrices of
	colony properties across time
	'''
	def __init__(self, analysis_config):
		self.colony_data_tracked_df = \
			analysis_config.get_colony_data_tracked_df()
		self.matrix_save_dir = \
			analysis_config.phase_col_properties_output_folder
		# identify indices of each unique timepoint and
		# unique_tracking_id value in their respective columns
		self._get_index_locations()
		# colony properties for which to NOT make property matrices
		cols_to_exclude = ['timepoint', 'phase', 'xy_pos_idx',
			'unique_tracking_id', 'main_image_name', 'bb_height', 'bb_width',
			'bb_x_left', 'bb_y_top']
		# colony properties for which to make property matrices
		self.cols_to_include = \
			list(set(self.colony_data_tracked_df.columns.to_list()) -
				set(cols_to_exclude))

	def _get_index_locations(self):
		'''
		Identifies indices of every unique value of timepoint and
		unique_tracking_id columns in self.colony_data_tracked_df, which
		can then be used to set those values in a numpy array with
		unique_tracking_id rows and timepoint columns
		'''
		self.timepoint_list, self.timepoint_indices = \
			np.unique(self.colony_data_tracked_df.timepoint.to_numpy(),
				return_inverse = True)
		self.unique_tracking_id_list, self.unique_tracking_id_indices = \
			np.unique(self.colony_data_tracked_df.unique_tracking_id.to_numpy(),
				return_inverse = True)
		# create a blank matrix of the right dimensions
		self.empty_col_property_mat = \
			np.empty((self.unique_tracking_id_list.shape[0],
				self.timepoint_list.shape[0]))
		self.empty_col_property_mat[:] = np.nan

	def _create_property_mat(self, col_property):
		'''
		Reads in dataframe of colony properties with information and
		organizes the data from the column corresponding to col_property
		into a matrix (as a pandas df) with timepoints as column names
		and unique_tracking_id as the row names
		'''
		# Indexing is ridiculous in pandas so fill in values in a numpy
		# array, then convert to pandas df and add column and row names
		# Get property values we're interested in as numpy array
		# Need to convert values to float, since only float-type numpy
		# arrays have np.nan values
		property_vals = \
			self.colony_data_tracked_df[col_property].to_numpy().astype(float)
		# create and populate colony property matrix
		col_property_mat = \
			self.empty_col_property_mat.copy().astype(property_vals.dtype)
		col_property_mat[self.unique_tracking_id_indices,
			self.timepoint_indices] = property_vals
		# convert to pandas df
		col_property_df = pd.DataFrame(col_property_mat,
			columns = self.timepoint_list, index = self.unique_tracking_id_list)
		# sort by rows and columns
		col_property_df.sort_index(inplace = True)
		col_property_df.reindex(sorted(col_property_df.columns), axis=1)
		return(col_property_df)

	def _save_property_df(self, col_property, col_property_df):
		'''
		Writes col_property_df to file
		'''
		write_path = os.path.join(self.matrix_save_dir,
			(col_property + '_property_mat.csv'))
		col_property_df.to_csv(write_path)

	def generate_imaging_info_df(self):
		'''
		Returns a dataframe of imaging info (e.g. xy position, phase)
		'''
		imaging_info_cols = ['unique_tracking_id', 'xy_pos_idx', 'phase']
		imaging_info_df = \
			self.colony_data_tracked_df[imaging_info_cols].drop_duplicates()
		# imaging_info_df should be indexed by unique_tracking_id
		imaging_info_df.set_index('unique_tracking_id', inplace = True)
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
		return(dict_for_gr['area'], dict_for_gr['time_in_seconds'], dict_for_gr['cX'],
			dict_for_gr['cY'])

class _DistanceCalculator(object):
	'''
	Calculates distance from each colony with a growth rate to nearest
	recognized colony (regardless of whether it was later filtered out)
	'''
	def __init__(self, cX_df, cY_df, analysis_config):
		self.colony_data_tracked_df = \
			analysis_config.get_colony_data_tracked_df()
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
		current_colonies = np.unique(current_data.unique_tracking_id)
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
	def __init__(self, areas, times_in_seconds, cX, cY, analysis_config,
				 imaging_info_df):
		self.analysis_config = analysis_config
		self.imaging_info_df = imaging_info_df
		self.unfilt_areas = areas
		# times nee
		self.unfilt_times = self._convert_times(times_in_seconds)
		self.unfilt_cX = cX
		self.unfilt_cY = cY
		self.columns_to_return = ['t0', 'gr', 'lag', 'rsq', 'foldX', 'mindist'] + \
				imaging_info_df.columns.to_list()

	def _convert_times(self, times_in_seconds):
		'''
		Converts times from absolute number of seconds to hours since
		1st image captured
		'''
		### !!! NEEDS UNITTEST
		rel_times_in_seconds = \
			times_in_seconds - self.analysis_config.first_timepoint
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
		self.filt_areas, self.pre_gr_removed_locations = \
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
		window_start_postions = \
			window_end_positions - \
			self.analysis_config.growth_window_timepoints + 1
		# create dataframe of positions to run linear regression on
		self.positions_for_regression = pd.DataFrame({
			't0': self.log_filt_areas.columns[window_start_postions],
			'start_col': window_start_postions,
			'range_stop_col': (window_start_postions +
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
		initial_log_areas = \
			self.log_filt_areas_mat[self.positions_for_regression.row, 0]
		self.positions_for_regression.lag = \
			(initial_log_areas - self.positions_for_regression.intercept) / \
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
			self.log_filt_areas.index[prefilt_growth_rate_data.row]
		# add vals for distance to closest neighboring colony to the df
		prefilt_growth_rate_data = self._apply_mindist(prefilt_growth_rate_data)
		# create filtration object
		post_growth_filter = \
			colony_filters.PostGrowthCombinedFilter(self.analysis_config,
				prefilt_growth_rate_data)
		# filter growth: set any value that needs to be removed in
		# post-growth filtration to NaN, and remove all-NaN rows
		self.filt_growth_rates, self.post_gr_removed_locations = \
			post_growth_filter.filter()

	def _select_colony_gr(self):
		'''
		Among filtered growth rates, selects the max calculated growth
		rate for each colony
		'''
		### NEEDS UNITTEST!!!
		# re-index self.filt_growth_rates, renaming current index to
		# 'unique_tracking_id', so that indices can be used below to
		# select maximal growth rates
		self.filt_growth_rates.rename_axis('unique_tracking_id',
			inplace = True )
		self.filt_growth_rates.reset_index(inplace = True)
		# identify locations of max growth rate for each colony id
		max_gr_idx = \
			self.filt_growth_rates.groupby('unique_tracking_id')['gr'].idxmax()
		filtered_gr = self.filt_growth_rates.loc[max_gr_idx]
		# re-index with unique_tracking_id
		filtered_gr.set_index('unique_tracking_id', inplace = True)
		# join filtered gr and imaging info gr by their indices
		combined_filtered_gr = filtered_gr.join(self.imaging_info_df)
		self.final_gr = combined_filtered_gr[self.columns_to_return]
		self.final_gr.to_csv(self.analysis_config.phase_gr_write_path)

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
		# select the highest growth rate for each colony to report, and
		# save results
		self._select_colony_gr()
		return(self.final_gr)

def run_growth_rate_analysis(analysis_config_file,
		repeat_image_analysis_and_tracking = True):
	'''
	Runs growth rate analysis from scratch (including image analysis
	steps) based on analysis_config_file
	If repeat_image_analysis_and_tracking is False, and phase colony
	properties file already exists, skip image analysis and tracking
	and go straight to growth rate assay
	'''
	analysis_config_obj_df = \
		analysis_configuration.set_up_analysis_config(analysis_config_file)
	for phase in analysis_config_obj_df.index:
		analysis_config = analysis_config_obj_df.at[phase, 'analysis_config']
		postphase_analysis_config = \
			analysis_config_obj_df.at[phase, 'postphase_analysis_config']
		# only perform image analysis and tracking steps if tracked
		# colony phase properties file is missing and/or
		# repeat_image_analysis_and_tracking is set to True
		if repeat_image_analysis_and_tracking or not \
			os.path.exists(analysis_config.phase_tracked_properties_write_path):
			track_colonies.track_single_phase_all_positions(analysis_config,
				postphase_analysis_config)
		measure_growth_rate(analysis_config)

def measure_growth_rate(analysis_config):
	'''
	Compiles data from individual timepoints into a single dataframe of
	tracked colonies, saves property matrices for each tracked colony,
	performs filtering and measures growth rates for all colonies that
	pass filtration
	'''
	colony_data_compiler = _CompileColonyData(analysis_config)
	area_mat_df, timepoint_mat_df, cX_mat_df, cY_mat_df = \
		colony_data_compiler.generate_property_matrices()
	imaging_info_df = colony_data_compiler.generate_imaging_info_df()
	growth_measurer = \
		_GrowthMeasurer(area_mat_df, timepoint_mat_df, cX_mat_df, cY_mat_df,
			analysis_config, imaging_info_df)
	gr_df = growth_measurer.find_growth_rates()
