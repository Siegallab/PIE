#!/usr/bin/python

'''
Classes for filtering colonies during growth rate analysis
Filter Classes must inherit from _FilterBaseClass and contain a
_filtration_method method that
	-	uses a parameter from self.analysis_config to filter
		self.df_to_filter
	-	return a boolean numpy array, the same size as the dataframe,
		with True at positions that will be kept after the filtration,
		and False at positions that will be removed
In some cases, the order of filtration matters (e.g. filtering by max
and min area should be done first, filtering by consecutive non-NaN area
window size should be done last), but ideally filters should be
independent of each other
'''

import numpy as np
import pandas as pd

class _FilterBaseClass(object):
	'''
	Generic class for performing filtration
	'''
	def __init__(self, df_to_filter, analysis_config):
		self.analysis_config = analysis_config
		self.df_to_filter = df_to_filter

	def _fill_in_row(self, bool_mat):
		'''
		Fills the rest of each row in bool_mat with True after the first
		True position
		'''
		# add a column of True values to the end of bool_mat so that argmax
		# function later on identifies last column as the final viable one
		# (rather than the 0th column, which would be incorrect)
		bool_mat_extend = \
			np.append(bool_mat,
				np.ones((bool_mat.shape[0],1), dtype = bool),
				axis = 1)
		# identify position of first True in each column
		first_true_position = np.argmax(bool_mat_extend, axis = 1)
		# For every row, every column after and including that row's
		# first_true_position needs to be true
		bool_mat_col_idx = np.arange(0, bool_mat.shape[1])
		output_mat = \
			bool_mat_col_idx >= np.transpose(first_true_position[np.newaxis])
		return(output_mat)

	def _filter_by_upper_change_bound(self, original_mat, upper_change_bound):
		'''
		Returns a filter which is false at any point in original_mat where
		the value change from the previous column in that row is greater
		than upper_change_bound
		'''
		mat_diff = np.diff(original_mat)
		# identify positions at which original_mat > upper_change_bound
		# need to ignore NAs, but also return 'False' where diff is NA
		with np.errstate(invalid='ignore'):
			pre_change_pos_bool = mat_diff > upper_change_bound
		# account for the fact that the columns where crash happens in
		# original_mat are the ones after where they are detected in the
		# diff matrix
		change_pos_bool = \
			np.append(np.zeros((pre_change_pos_bool.shape[0],1), dtype = bool),
				pre_change_pos_bool,
				axis = 1)
		change_pos_bool_filled = self._fill_in_row(change_pos_bool)
		# positions where the crash happens and all the ones after don't
		# pass the filter
		filter_pass = np.invert(change_pos_bool_filled)
		return(filter_pass)

	def _propagate_filter_across_columns(self, filter_pass_rows):
		''' Repeat filter_pass_rows in every column '''
		if not isinstance(filter_pass_rows, np.ndarray):
			raise TypeError(
				'_propagate_filter_across_columns only accepts numpy arrays'
				)
		col_num = len(self.df_to_filter.columns)
		filter_pass = \
			filter_pass_rows[np.newaxis].T.repeat(col_num, axis = 1)
		return(filter_pass)

	def _id_removed_locations(self, filter_pass_bool):
		'''
		Returns a pandas df of indices and column names from
		self.df_to_filter indicating the first column at which any
		filtered row is filtered
		'''
		# TODO: Consider returning the full list of columns filtered out in
		# each row
		if filter_pass_bool.size == 0:
			output_df = pd.DataFrame(columns = ['filtered_columns'],
				index = [])
		else:
			filter_fail_bool = np.invert(filter_pass_bool)
			filtered_rows = filter_fail_bool.any(1)
			filter_fail_bool_filtered_rows = filter_fail_bool[filtered_rows]
			first_true_col = np.argmax(filter_fail_bool_filtered_rows, axis = 1)
			filtered_indices = self.df_to_filter.index[filtered_rows]
			filtered_columns = self.df_to_filter.columns[first_true_col]
			output_df = pd.DataFrame({'filtered_columns': filtered_columns},
				index = filtered_indices)
		return(output_df)

	def _filtration_method(self):
		'''
		Class-specific method that performs filtration
		Returns boolean np array with True at positions that pass filter
		'''
		pass

	def filter_data(self):
		'''
		Runs class-specifc _filtration_method
		Returns boolean np array with True at positions that pass filter
		and a list of index-column name tuples from self.df_to_filter
		'''
		if self.df_to_filter.empty:
			filter_pass = self.df_to_filter.to_numpy().astype(bool)
		else:
			filter_pass = self._filtration_method()
			if not isinstance(filter_pass, np.ndarray):
				raise TypeError('_filtration_method must return a numpy array')
			elif not filter_pass.dtype == 'bool':
				raise TypeError(
					'_filtration_method must return a bool-type numpy array')
		removed_locations = self._id_removed_locations(filter_pass)
		return(filter_pass, removed_locations)

class _FilterByMinArea(_FilterBaseClass):
	'''
	Filters out colonies whose area is below min_colony_area (at that
	timepoint)
	'''
	def _filtration_method(self):
		### !!! NEEDS UNITTEST
		# identify positions at which area too small
		# need to ignore NAs, but also return 'False' where diff is NA
		with np.errstate(invalid='ignore'):
			areas_too_small = self.df_to_filter.to_numpy() < \
				self.analysis_config.min_colony_area
		filter_pass = np.invert(areas_too_small)
		return(filter_pass)

class _FilterByMaxArea(_FilterBaseClass):
	'''
	Filters out colonies whose area is above max_colony_area (at that
	timepoint)
	Returns boolean np array with True at positions that pass filter
	'''
	def _filtration_method(self):
		### !!! NEEDS UNITTEST
		# identify positions at which area too large
		# need to ignore NAs, but also return 'False' where diff is NA
		with np.errstate(invalid='ignore'):
			areas_too_small = self.df_to_filter.to_numpy() > \
				self.analysis_config.max_colony_area
		filter_pass = np.invert(areas_too_small)
		return(filter_pass)

class _FilterByMaxAreaPixelDecrease(_FilterBaseClass):
	'''
	Filters out colonies that decrease in area by more than
	max_area_pixel_decrease pixels over a single timepoint after the
	decrease happens
	Returns boolean np array with True at positions that pass filter
	Equivalent to filtration by GR.crash.pixels in previous versions
	'''
	def _filtration_method(self):
		# this problem can be restated as not allowing the negative area
		# to increase more than max_area_pixel_decrease
		upper_change_bound = self.analysis_config.max_area_pixel_decrease
		mat_to_filter_by_upper_change = -self.df_to_filter.to_numpy()
		filter_pass = \
			self._filter_by_upper_change_bound(mat_to_filter_by_upper_change,
				upper_change_bound)
		return(filter_pass)

class _FilterByMaxAreaFoldDecrease(_FilterBaseClass):
	'''
	Filters out colonies that decrease in area by more than
	max_area_fold_decrease-fold over a single timepoint after the
	decrease happens
	Returns boolean np array with True at positions that pass filter
	Equivalent to lower-side filtration by GR.crash.fold in previous
	versions
	'''
	def _filtration_method(self):
		# this problem can be restated as not allowing negative log area to
		# increase more than log(max_area_fold_decrease)
		# ignore errors but treat log(0) as -inf
		with np.errstate(divide='ignore'):
			upper_change_bound = \
				np.log(self.analysis_config.max_area_fold_decrease)
			mat_to_filter_by_upper_change = \
				-np.log(self.df_to_filter.to_numpy())
		filter_pass = \
			self._filter_by_upper_change_bound(mat_to_filter_by_upper_change,
				upper_change_bound)
		return(filter_pass)

class _FilterByMaxAreaFoldIncrease(_FilterBaseClass):
	'''
	Filters out colonies that increase in area by more than
	max_area_fold_increase-fold over a single timepoint after the
	increase happens
	Returns boolean np array with True at positions that pass filter
	Similar to upper-side filtration by GR.crash.fold in previous
	versions
	'''
	def _filtration_method(self):
		# this problem can be restated as not allowing log area to increase
		# more than log(max_area_fold_decrease)
		# ignore errors but treat log(0) as -inf
		with np.errstate(divide='ignore'):
			upper_change_bound = \
				np.log(self.analysis_config.max_area_fold_increase)
			mat_to_filter_by_upper_change = np.log(self.df_to_filter.to_numpy())
		filter_pass = \
			self._filter_by_upper_change_bound(mat_to_filter_by_upper_change,
				upper_change_bound)
		return(filter_pass)

class _FilterByColonyAppearanceTime(_FilterBaseClass):
	'''
	Filters out any colonies that appear after the first timepoint
	'''
	def _filtration_method(self):
		### !!! NEEDS UNITTEST
		timepoints = self.df_to_filter.columns.to_numpy()
		valid_timepoint_areas = self.df_to_filter[np.min(timepoints)]
		# identify colonies that have an area recorded at the
		# first timepoint
		filter_pass_rows = \
			valid_timepoint_areas.notnull().to_numpy()
		# repeat filter_pass_rows in every column
		filter_pass = self._propagate_filter_across_columns(filter_pass_rows)
		return(filter_pass)

class _FilterByMinGrowthTime(_FilterBaseClass):
	'''
	Filters out any colonies whose areas haven't been measured in at
	least the number of frames provided by minimum_growth_time
	'''
	# TODO: Consider removing this filter; it's redundant with the new
	# way of filtering by 
	def _filtration_method(self):
		### !!! NEEDS UNITTEST
		tracked_timepoint_number = np.sum(self.df_to_filter.notnull(), axis = 1)
		filter_pass_rows = tracked_timepoint_number.to_numpy() >= \
			self.analysis_config.minimum_growth_time
		filter_pass = self._propagate_filter_across_columns(filter_pass_rows)
		return(filter_pass)

class _FilterByGrowthWindowTimepoints(_FilterBaseClass):
	'''
	Filters out any colony timepoints that are not part of a consecutive
	set of non-NaN growth_window_timepoints measurements for that colony
	'''
	def _filtration_method(self):
		### !!! NEEDS UNITTEST
		# initialize filter_pass
		filter_pass = np.zeros(self.df_to_filter.shape, dtype = bool)
		# if self.analysis_config.growth_window_timepoints greater than
		# the number of timepoints, filter everything
		if self.analysis_config.growth_window_timepoints > \
			self.df_to_filter.shape[1]:
			return(filter_pass)
		# identify non-NaN positions in areas
		areas_not_null = self.df_to_filter.notnull().to_numpy()
		# identify column positions where stretches of non-null values
		# start (non_null_start_col), the column after the end
		# (non_null_stop_cols), and the corresponding row for each of
		# those (non_null_row)
		# These three vectors are 1-D vectors of the same length
		areas_not_null_diff = np.diff(areas_not_null, prepend = 0, append = 0)
		non_null_row, non_null_start_col = np.where(areas_not_null_diff == 1)
		_, non_null_stop_col = np.where(areas_not_null_diff == -1)

		# Identify positions in the above non_null_... vectors that
		# correspond to large enough windows for growth rate calculation
		non_null_stretch_length = non_null_stop_col - non_null_start_col
		window_filter_pass_idx = non_null_stretch_length >= \
			self.analysis_config.growth_window_timepoints
		if any(window_filter_pass_idx):
			# Subset only start and stop columns, as well as row numbers,
			# that correspond to large enough windows for growth rate
			# calculation
			allowed_start_cols = non_null_start_col[window_filter_pass_idx]
			allowed_stop_cols = non_null_stop_col[window_filter_pass_idx]
			allowed_rows = non_null_row[window_filter_pass_idx]
			# create 1-D vectors of column and row positions of each
			# colony that is part of a consecutive window of at least
			# growth_window_timepoints non-NaN areas
			allowed_col_positions = \
				vectorized_arange(allowed_start_cols,
					allowed_stop_cols).astype(int)
			allowed_row_positions = \
				np.repeat(allowed_rows,
					(allowed_stop_cols - allowed_start_cols))
			# set positions of passing colonies to true
			filter_pass[allowed_row_positions, allowed_col_positions] = True
		return(filter_pass)

class _FilterByMinFoldX(_FilterBaseClass):
	'''
	Filters out any colony growth rates that didn't increase by at least
	min_foldX fold at some point over the course of the growth window
	'''
	def _filtration_method(self):
		### !!! NEEDS UNITTEST
		filter_pass_rows = self.df_to_filter.foldX.to_numpy() >= \
			self.analysis_config.min_foldX
		# repeat filter_pass_rows in every column
		filter_pass = self._propagate_filter_across_columns(filter_pass_rows)
		return(filter_pass)

class _FilterByMinCorrelation(_FilterBaseClass):
	'''
	Filters out any colony growth rates that didn't have an r-squared
	value of at least min_correlation
	'''
	def _filtration_method(self):
		### !!! NEEDS UNITTEST
		filter_pass_rows = \
			self.df_to_filter.rsq.to_numpy() >= \
				self.analysis_config.min_correlation
		# repeat filter_pass_rows in every column
		filter_pass = self._propagate_filter_across_columns(filter_pass_rows)
		return(filter_pass)

class _FilterByMinNeighborDist(_FilterBaseClass):
	'''
	Filters out any colonies that are closer than min_neighbor_dist to
	their nearest neighbor
	'''
	def _filtration_method(self):
		### !!! NEEDS UNITTEST
		filter_pass_rows = \
			self.df_to_filter.mindist.to_numpy() >= \
				self.analysis_config.min_neighbor_dist
		# repeat filter_pass_rows in every column
		filter_pass = self._propagate_filter_across_columns(filter_pass_rows)
		return(filter_pass)

class _FilterByMinEdgeDist(_FilterBaseClass):
	'''
	Filters out any colonies that are closer than min_neighbor_dist/2
	to the image edge
	'''
	def _filtration_method(self):
		### !!! NEEDS UNITTEST
		min_edge_dist = self.analysis_config.min_neighbor_dist/2
		# remember that im_size is number of rows and columns, but
		# indexing starts at 0
		max_y = self.analysis_config.im_height-1-min_edge_dist
		min_y = min_edge_dist
		max_x = self.analysis_config.im_width-1-min_edge_dist
		min_x = min_edge_dist
		y_pass_rows = np.logical_and(
			self.df_to_filter.cym.to_numpy() <= max_y,
			self.df_to_filter.cym.to_numpy() >= min_y
			)
		x_pass_rows = np.logical_and(
			self.df_to_filter.cxm.to_numpy() <= max_x,
			self.df_to_filter.cxm.to_numpy() >= min_x
			)
		filter_pass_rows = np.logical_and(x_pass_rows, y_pass_rows)
		# repeat filter_pass_rows in every column
		filter_pass = self._propagate_filter_across_columns(filter_pass_rows)
		return(filter_pass)

class CombinedFilterBaseClass(object):
	'''
	Parent class for filter classes that will perform filtration on
	identified colonies before and after growth rate analysis
	'''
	def __init__(self, analysis_config, df_to_filter):
		self.analysis_config = analysis_config
		self.df_to_filter = df_to_filter
		# initialize dataframe of colonies removed by filtration
		self.removed_colonies_df = pd.DataFrame()
		# TODO: It's worth rethinking a lot of the filtration steps we're performing

	def _run_filtration(self, filtration_class, filtration_name, df_to_filter,
		combined_filter_pass_bool):
		'''
		Runs filtration of df_to_filter using object of filtration_class
		Updates and returns combined_filter_pass_bool
		'''
		### !!! NEEDS UNITTEST
		# create filtration object
		filtration_obj = filtration_class(df_to_filter, self.analysis_config)
		# perform filtering
		filter_pass_bool, removed_locations = filtration_obj.filter_data()
		# update combined_filter_pass_bool
		combined_filter_pass_bool = \
			np.logical_and(combined_filter_pass_bool, filter_pass_bool)
		# update dataframe of removed colonies with current
		# filtration step
		removed_locations.columns = [filtration_name]
		self.removed_colonies_df = \
			self.removed_colonies_df.join(removed_locations, how = 'outer')
		return(combined_filter_pass_bool)

	def _filter_df_by_filter_pass_mat(self, df_to_filter, filter_pass_mat):
		'''
		Returns filtered_df, which is df_to_filter but with any
		positions that are False in filter_pass_mat set to np.nan
		'''
		### !!! NEEDS UNITTEST
		filtered_mat = df_to_filter.copy().to_numpy()
		filtered_mat[np.invert(filter_pass_mat)] = np.nan
		filtered_df = pd.DataFrame(filtered_mat, index = df_to_filter.index,
			columns = df_to_filter.columns)
		return(filtered_df)

	def _filter_from_filtration_dict(self, filtration_dict, df_to_filter,
		combined_filter_pass_bool):
		'''
		Runs filter objects in filtration_dict on df_to_filter, returns
		filtered df_to_filter and updated combined_filter_pass_bool
		'''
		### !!! NEEDS UNITTEST
		for filtration_type, filtration_class in filtration_dict.items():
			# perform filtration, update combined_filter_pass_bool, add
			# removed colony locations to self.removed_colonies_df
			combined_filter_pass_bool = \
				self._run_filtration(filtration_class, filtration_type,
					df_to_filter, combined_filter_pass_bool)
		# get df_to_filter w/ NaN in positions that don't pass filtering
		filtered_df = self._filter_df_by_filter_pass_mat(df_to_filter,
			combined_filter_pass_bool)
		return(filtered_df, combined_filter_pass_bool)

	def filter(self):
		'''
		Defines filtration dictionaries and filters df_to_filter using
		the filtration classes inside filtration dictionaries
		Creates self.combined_filter_pass_bool that can be used to
		reproduce the original filter on a different dataframe
		Returns filtered_df
		'''
		pass

	def reproduce_filter(self, new_df):
		'''
		Filters new_df using the same filtration bool matrix that was
		created during the original filter step, setting positions that
		failed filtering based on self.df_to_filter to np.nan in new_df
		'''
		### !!! NEEDS UNITTEST
		new_df_filtered = self._filter_df_by_filter_pass_mat(new_df,
			self.combined_filter_pass_bool)
		new_df_filtered.dropna(how = 'all', inplace = True)
		return(new_df_filtered)

class PreGrowthCombinedFilter(CombinedFilterBaseClass):
	'''
	Performs filtration on colonies before growth rate analysis
	'''

	def filter(self):
		'''
		Runs all filtration necessary before growth rate analysis
		Defines filtration dictionaries and filters df_to_filter using
		the filtration classes inside filtration dictionaries
		First runs filtration by min_colony_area and max_colony_area, then by area
		increases and decreares over time, and finally by consecutive
		growth_window_timepoints (order here may matter in a
		small number of cases)
		Returns filtered_df
		'''
		### !!! NEEDS UNITTEST
		# specify the two filration dictionaries that will be used
		pre_gr_filtration_dict_round_1 = {
			'min_colony_area': _FilterByMinArea,
			'max_colony_area': _FilterByMaxArea
			}
		pre_gr_filtration_dict_round_2 = {
			'colony_appearance_time': _FilterByColonyAppearanceTime,
			'min_growth_time': _FilterByMinGrowthTime,
			'max_area_fold_increase': _FilterByMaxAreaFoldIncrease,
			'max_area_fold_decrease': _FilterByMaxAreaFoldDecrease,
			'max_area_pixel_decrease': _FilterByMaxAreaPixelDecrease
			}
		pre_gr_filtration_dict_round_3 = {
			'growth_window_timepoints': _FilterByGrowthWindowTimepoints
			}
		# df_to_filter is unfilt_areas
		unfilt_areas = self.df_to_filter
		# initialize bool of pre-gr colonies passing filter
		combined_filter_pass_bool = \
			np.ones(unfilt_areas.shape, dtype = bool)
		# run filtration with first set of filters
		part_1_filtered_areas, combined_filter_pass_bool = \
			self._filter_from_filtration_dict(pre_gr_filtration_dict_round_1,
				unfilt_areas, combined_filter_pass_bool)
		# run filtration with second set of filters
		part_2_filtered_areas, combined_filter_pass_bool = \
			self._filter_from_filtration_dict(pre_gr_filtration_dict_round_2,
				part_1_filtered_areas, combined_filter_pass_bool)
		# run filtration with third set of filters
		self.filtered_areas, self.combined_filter_pass_bool = \
			self._filter_from_filtration_dict(pre_gr_filtration_dict_round_3,
				part_1_filtered_areas, combined_filter_pass_bool)
		# remove rows containing all NaN from self.filtered_areas
		self.filtered_areas.dropna(how = 'all', inplace = True)
		return(self.filtered_areas, self.removed_colonies_df)


class PostGrowthCombinedFilter(CombinedFilterBaseClass):
	'''
	Performs filtration on colonies after growth rate analysis
	'''

	def filter(self):
		'''
		Runs all filtration necessary after growth rate analysis
		Defines filtration dictionaries and filters df_to_filter using
		the filtration classes inside filtration dictionaries
		Runs filtration by min_foldX and min_correlation
		Returns filtered_df
		'''
		### !!! NEEDS UNITTEST
		# specify the two filration dictionaries that will be used
		post_gr_filtration_dict = {
			'min_foldX': _FilterByMinFoldX,
			'min_correlation': _FilterByMinCorrelation,
			'min_neighbor_dist': _FilterByMinNeighborDist,
			'min_edge_dist': _FilterByMinEdgeDist
			}
		# df_to_filter is unfilt_areas
		unfilt_growth_rates = self.df_to_filter
		# initialize bool of pre-gr colonies passing filter
		combined_filter_pass_bool = \
			np.ones(unfilt_growth_rates.shape, dtype = bool)
		# run filtration
		self.filtered_growth_rates, self.combined_filter_pass_bool = \
			self._filter_from_filtration_dict(post_gr_filtration_dict,
				unfilt_growth_rates, combined_filter_pass_bool)
		# remove rows containing all NaN from self.filtered_areas
		self.filtered_growth_rates.dropna(how = 'all', inplace = True)
		# self.removed_colonies_df in post-gr analysis can have
		# duplicates due to multiple growth rate calculations per
		# colony being present in self.df_to_filter; drop those
		# NB: can't use drop_duplicates() because ir ignores index
		self.removed_colonies_df = \
			self.removed_colonies_df[
				~self.removed_colonies_df.reset_index().duplicated().values
				]
		# change non-NA values in self.removed_colonies_df to make
		# output more sensical
		self.removed_colonies_df[self.removed_colonies_df.notnull()] = 'all'
		self.removed_colonies_df.min_correlation[
			self.removed_colonies_df.min_correlation.notnull()
			] = 'some or all candidate windows'
		return(self.filtered_growth_rates, self.removed_colonies_df)


def vectorized_arange(range_start_array, range_stop_array):
	'''
	Vectorized function that creates 1-D np array holding the equivalent
	of arrays np.arange(range_start_array[i], range_stop_array[i]) for
	every i consecutively
	Assumes step = 1
	'''
	# length of each positions arange output
	subarray_lengths = range_stop_array - range_start_array
	# set up array of startpoint values only of the right length
	start_array = np.repeat(range_start_array, subarray_lengths)
	# set up array that will produce arange_result when added to
	# start_array
	# find positions of starts of new subarrays in array_result
	subarray_start_positions = \
		np.append([0], np.cumsum(subarray_lengths[0:-1]))
	# identify cumulative sum of True values corresponding to position
	# of each non-overlapping subarray
	cum_sum_subarrays = cum_sum_since_subarray_start(start_array.shape[0],
		subarray_start_positions, subarray_lengths)
	array_result = start_array + cum_sum_subarrays - 1
	return(array_result)

def cum_sum_since_subarray_start(result_length, subarray_start_positions,
	subarray_lengths):
	'''
	Given starts and lengths of non-overlapping subarrays of 1s in a 1-D
	array, creates an array of result_length that contains the
	cumulative sums since the start of each subarray
	'''
	### !!! NEEDS UNITTEST for overlapping part
	# check that subarrays non-overlapping
	start_indices_order = np.argsort(subarray_start_positions)
	sorted_start_positions = subarray_start_positions[start_indices_order]
	start_sorted_lengths = subarray_lengths[start_indices_order]
	start_sorted_end_positions = sorted_start_positions + start_sorted_lengths-1
	if any([any(start_sorted_end_positions[(idx+1):] < start) for
		idx, start in enumerate(sorted_start_positions)]):
		raise ValueError('Subarrays are overlapping')
	# initialize naive increase at each position relative to the previous one
	positionwise_increase = np.append([1], np.ones(result_length-1))
	# find length of each previous subarray for subarray_start_positions
	prev_subarray_length = np.append([0], subarray_lengths[0:-1])
	# identify length of arrays between subarrays
	inter_subarray_length = subarray_start_positions - \
		(prev_subarray_length + np.append([0], subarray_start_positions[:-1]))
	# subtract length of previous subarray from every subarray start
	# position in positionwise_increase to reset this value to 0 in the
	# cumulative sum of positionwise_increase
	positionwise_increase[subarray_start_positions] = \
		positionwise_increase[subarray_start_positions] - \
			prev_subarray_length - inter_subarray_length
	cumulative_sum_subarrays = np.cumsum(positionwise_increase)
	return(cumulative_sum_subarrays)
