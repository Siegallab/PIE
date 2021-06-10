#!/usr/bin/python

'''
Processes properties of colonies tracked through time
'''

import numpy as np
import pandas as pd

class CompileColonyData(object):
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
		dict_for_gr = dict.fromkeys(properties_to_return, pd.DataFrame())
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
		tp_to_use = tp_to_use_idx.copy()
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
			nan_mat = pd.isnull(col_property_mat_df_subset).to_numpy()
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
					'col_property_mat_df; instead, timepoint_label is ' + str(timepoint_label))
			tp_to_use = col_property_mat_df_subset.columns[tp_to_use_idx]
		output_vals = \
			col_property_mat[np.arange(0,col_property_mat.shape[0]), tp_to_use_idx]
	return(output_vals, tp_to_use)

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