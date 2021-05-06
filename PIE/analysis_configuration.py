#!/usr/bin/python

'''
Tracks colonies through time in a single imaging field
'''

import cv2
import numpy as np
import os
import warnings
import pandas as pd
from PIL import Image

# load dataframe of parameters and descriptions
PIE_package_path = os.path.abspath(os.path.dirname(__file__))
parameter_file = os.path.join(
	PIE_package_path, '..', 'PIE_data', 'param_descriptions.csv'
	)
param_description_df = pd.read_csv(parameter_file)

# list fields that must be specified in analysis config
required_fields_general = \
	['fluor_channel_scope_labels', 'fluor_channel_names',
	'fluor_channel_thresholds', 'fluor_channel_timepoints',
	'timepoint_spacing', 'hole_fill_area',
	'cleanup', 'perform_registration', 'max_proportion_exposed_edge', 'input_path',
	'output_path', 'im_file_extension', 'label_order_list',
	'total_xy_position_num', 'first_timepoint', 'total_timepoint_num',
	'first_xy_position', 'extended_display_positions',
	'timepoint_label_prefix', 'position_label_prefix',
	'main_channel_label', 'main_channel_imagetype', 'im_format',
	'parent_phase', 'max_area_pixel_decrease',
	'max_area_fold_decrease', 'max_area_fold_increase',
	'min_colony_area', 'max_colony_area', 'min_correlation',
	'min_foldX', 'minimum_growth_time', 'growth_window_timepoints',
	'min_neighbor_dist', 'max_colony_num']

required_fields_minimal = \
	['fluor_channel_scope_labels', 'fluor_channel_names',
	'fluor_channel_thresholds', 'fluor_channel_timepoints',
	'input_path', 'first_xy_position', 'extended_display_positions',
	'timepoint_label_prefix',
	'output_path', 'im_file_extension', 'label_order_list',
	'total_xy_position_num',
	'position_label_prefix',
	'im_format',
	'parent_phase']

class _ImageRetriever(object):
	'''
	Retrieves image for current position, channel, timepoint
	'''

	def get_image(self, timepoint, position, channel_label):
		'''
		Returns single cv2 image for current timepoint, position, and
		channel_label
		'''
		pass

class _IndivImageRetriever(_ImageRetriever):
	'''
	Retrieves image for current position, channel, timepoint for images
	saved as individual files for every timepoint/color
	'''

	def get_image(self, **kwargs):
		'''
		Returns single cv2 image from current im_filepath
		(Doesn't use timepoint or channel information passed to it)
		'''
		### !!! NEEDS UNITTEST
		image = cv2.imread(kwargs['im_filepath'], cv2.IMREAD_ANYDEPTH)
			# cv2.imread returns none if im_filepath doesn't exist
		return(image)
		

class _StackImageRetriever(_ImageRetriever):
	'''
	Retrieves image for current position, channel, timepoint for images
	saved as stack of images for every timepoint/color
	'''
	#TODO: Integrate micromanager and NIS elements imagestacks - currently can only read data saved as individual tifs
	def __init__(self, im_path):
		pass

	def get_image(self, **kwargs):
		'''
		Returns single cv2 image for current timepoint, position, and
		channel_label
		'''
		pass

class MinimalAnalysisConfig(object):
	'''
	Handles experimental configuration details in for experiments 
	without timepoints or main channel
	'''
	def __init__(self, phase_num, input_path, output_path, im_file_extension,
		label_order_list, total_xy_position_num, position_label_prefix,
		fluor_channel_df, im_format, extended_display_positions,
		timepoint_label_prefix, xy_position_vector):
		'''
		Reads setup_file and creates analysis configuration
		'''
		# max timepoint number
		self.total_timepoint_num = 1
		# specify phase
		self.phase_num = phase_num
		# max xy position label
		self.total_xy_position_num = int(total_xy_position_num)
		# path of input images
		self.input_path = input_path
		# path of output image folder
		self.output_path = output_path
		# file extension of input images
		self.im_file_extension = im_file_extension
		# save order in which time, position, channel labels are listed,
		# after checking that it contains the necessary info
		# note that timepoint still required in this list even though not used
		if set(label_order_list) == set(['timepoint', 'channel', 'position']):
			self.label_order_list = label_order_list
		else:
			raise ValueError(
				'Label order list must consist of timepoint, channel, and ' +
				'position, even if not all these are used')
		# set up folder to save outputs
		self._create_output_paths()
		# set up list of possible xy positions
		self.xy_position_vector = xy_position_vector
		# labels used for xy position
		self.position_label_prefix = position_label_prefix
		# find size of images
		self._find_im_size()
		self.fluor_channel_df = fluor_channel_df
			# column names: fluor_channel_label,
			# fluor_channel_column_name, fluor_threshold
		# set up image retriever depending on im_format
		if im_format == 'individual':
			self.image_retriever = _IndivImageRetriever()
		else:
			raise ValueError('image format ' + im_format + ' not recognized')
		self.extended_display_positions = extended_display_positions
		# labels used for timepoint number 
		self.timepoint_label_prefix = timepoint_label_prefix

	def __eq__(self, other):
		identical_vars = self.__dict__ == other.__dict__
		identical_class = self.__class__ == other.__class__
		return(identical_vars and identical_class)

	def _create_output_paths(self):
		'''
		Creates output paths for phase data, and sets output paths for
		growth rate and colony property dataframes
		'''
		self._create_phase_output()
		self.combined_gr_write_path = \
			os.path.join(self.output_path, 'growth_rates_combined.csv')
		self.combined_tracked_properties_write_path = \
			os.path.join(self.output_path, 'colony_properties_combined.csv')
		self.col_properties_output_folder = \
			os.path.join(self.output_path, 'positionwise_colony_properties')
		self.movie_folder = \
			os.path.join(self.output_path, 'movies')
		try:
			os.makedirs(self.col_properties_output_folder)
		except:
			pass

	def _create_phase_output(self):
		'''
		Creates folder for results of colony properties across phases,
		as well as within-phase results, if they don't already exist
		'''
		# NEED UNITTEST FOR JUST THIS METHOD?
		self.phase_output_path = os.path.join(self.output_path,
			('phase_' + str(self.phase_num)))
		try:
			os.makedirs(self.phase_output_path)
		except:
			pass
		self.phase_col_property_mats_output_folder = \
			os.path.join(self.phase_output_path,
				'positionwise_colony_property_matrices')
		try:
			os.makedirs(self.phase_col_property_mats_output_folder)
		except:
			pass
		# create filename for growth rate output file for current phase
		self.phase_gr_write_path = \
			os.path.join(self.phase_output_path, 'growth_rates.csv')
		# create filename for filtered colony output file for current
		# phase
		self.filtered_colony_file = \
			os.path.join(self.phase_output_path, 'filtered_colonies.csv')

	def _find_im_size(self):
		'''
		Assumes all images are the same size
		'''
		# find image files in self.input_path
		im_files = [
			f for f in os.listdir(self.input_path)
			if f.endswith(self.im_file_extension)
			]
		# open *some* input image
		im_to_use = im_files[0]
		self.size_ref_im = im_to_use
		# NB: Pillow doesn't open jpegs saved through matlab with weird
		# bitdepths, i.e. in old matlab PIE code
		with Image.open(os.path.join(self.input_path,im_to_use)) as im:
			self.im_width, self.im_height = im.size

	def _reformat_values(self, int_to_format, max_val_num):
		'''
		Returns int_to_format as string, padded with 0s to match the
		number of digits in max_val_num
		If int_to_format is None, returns empty string
		'''
		### !!! NEEDS UNITTEST
		if int_to_format is None:
			formatted_string = ''
		else:
			digit_num = np.ceil(np.log10(max_val_num+1)).astype(int)
			formatted_string = '{:0>{}d}'.format(int_to_format, digit_num)
		return(formatted_string)

	def _generate_filename(self, timepoint, position, channel_label):
		'''
		Returns filename for image file given timepoint, position,
		channel_label, as well as its image label (filename without
		extension)
		'''
		### !!! NEEDS UNITTEST
		im_label = self.create_file_label(timepoint, position, channel_label)
		im_filepath = os.path.join(self.input_path, im_label + '.' + 
			self.im_file_extension)
		return(im_filepath, im_label)

	def create_file_label(self, timepoint, position, channel_label):
		'''
		Creates label for image filename, concatenating formatted
		timepoint, xy position, and provided channel label in the
		correct order
		'''
		### !!! NEEDS UNITTEST
		current_timepoint_str = self.timepoint_label_prefix + \
			self._reformat_values(timepoint, self.total_timepoint_num)
		current_position_str = self.position_label_prefix + \
			self._reformat_values(position, self.total_xy_position_num)
		current_point_label_dict = \
			{'timepoint': current_timepoint_str,
			'channel': channel_label,
			'position': current_position_str}
		file_label = ''
		# loop through ordered list of labels and append correct info to
		# filename one-by-one
		for label_key in self.label_order_list:
			current_label = current_point_label_dict[label_key]
			# current label may be np.nan if channel not specified
			if isinstance(current_label, str) and current_label != '':
				file_label = file_label + current_point_label_dict[label_key]
		return(file_label)

	def get_image(self, timepoint, channel):
		'''
		Returns an image at current xy position for timepoint and
		channel, as well as the image's 'image label' (filename without
		extension) and the time (in seconds) at which it was taken
		'''
		### !!! NEEDS UNITTEST
		im_filepath, im_label = \
			self._generate_filename(timepoint, self.xy_position_idx, channel)
		image = self.image_retriever.get_image(im_filepath = im_filepath,
			timepoint = timepoint, channel = channel)
		# get image time
		if image is None or timepoint is None:
			image_time = None
		else:
			# if timepoint dict exists, get time value from there;
			# otherwise, get it from the file modification date
			if self.timepoint_dict:
				image_time = self.timepoint_dict[timepoint]
			else:
				image_time = os.path.getmtime(im_filepath)
		return(image, im_label, image_time)

	def set_xy_position(self, xy_position_idx):
		'''
		Sets the xy position to be used by the analysis config
		'''
		### !!! NEEDS UNITTEST
		if xy_position_idx not in self.xy_position_vector:
			raise IndexError('Unexpected xy position index ' + str(xy_position_idx) +
				' in phase ' + str(self.phase_num))
		# current position being imaged
		self.xy_position_idx = xy_position_idx
		# determine wether non-essential info (threshold plot outputs,
		# boundary images, etc) need to be saved for this experiment
		self.save_extra_info = \
			xy_position_idx in self.extended_display_positions
		# create filename for tracked colony properties output file for
		# current xy position
		self.tracked_properties_write_path = \
			os.path.join(self.col_properties_output_folder,
				'xy_' + str(xy_position_idx) + 
				'_col_props_with_tracking_pos.parquet')

class AnalysisConfig(MinimalAnalysisConfig):
	'''
	Handles experimental configuration details
	'''
	def __init__(self, phase_num, hole_fill_area, cleanup, perform_registration,
		max_proportion_exposed_edge, input_path, output_path, im_file_extension,
		label_order_list, total_xy_position_num, first_timepoint,
		total_timepoint_num, timepoint_spacing, timepoint_label_prefix,
		position_label_prefix, main_channel_label, main_channel_imagetype,
		fluor_channel_df, im_format, extended_display_positions,
		xy_position_vector, minimum_growth_time,
		growth_window_timepoints, max_area_pixel_decrease,
		max_area_fold_decrease, max_area_fold_increase, min_colony_area,
		max_colony_area, min_correlation, min_foldX, min_neighbor_dist, max_colony_num):
		'''
		Reads setup_file and creates analysis configuration
		'''
		super(AnalysisConfig, self).__init__(
			phase_num, input_path, output_path, im_file_extension,
			label_order_list, total_xy_position_num, position_label_prefix,
			fluor_channel_df, im_format, extended_display_positions,
			timepoint_label_prefix, xy_position_vector
			)
		# max timepoint number
		self.total_timepoint_num = int(total_timepoint_num)
		# specify image analysis parameters
		self.hole_fill_area = float(hole_fill_area)
		self.cleanup = bool(cleanup)
		self.perform_registration = bool(perform_registration)
		self.max_proportion_exposed_edge = float(max_proportion_exposed_edge)
		# specify growth rate analysis parameters
		self.minimum_growth_time = int(minimum_growth_time)
		growth_window_timepoint_int = int(growth_window_timepoints)
		if growth_window_timepoint_int == 0:
			self.growth_window_timepoints = self.total_timepoint_num
		else:
			self.growth_window_timepoints = growth_window_timepoint_int
		self.max_area_pixel_decrease = float(max_area_pixel_decrease)
		self.max_area_fold_decrease = float(max_area_fold_decrease)
		self.max_area_fold_increase = float(max_area_fold_increase)
		self.min_colony_area = float(min_colony_area)
		self.max_colony_area = float(max_colony_area)
		self.min_correlation = float(min_correlation)
		self.min_foldX = float(min_foldX)
		self.min_neighbor_dist = float(min_neighbor_dist)
		self.max_colony_num = float(max_colony_num)
		# set up dictionary of timepoint times
		self._set_up_timevector(timepoint_spacing, first_timepoint)
		# find time of first existing file
		self._find_first_timepoint()
		# specify type of image (brightfield or phase_contrast) is in
		# the main channel
		self.main_channel_imagetype = main_channel_imagetype
		# set up channel labels
		self.main_channel_label = main_channel_label
		self._run_parameter_tests()

	def _set_up_timevector(self, timepoint_spacing, first_timepoint):
		'''
		Sets up a dictionary of timepoint times, in seconds, at which
		images were taken, if such information is provided
		timepoint spacing can be:
			a vector of seconds at each timepoint
			a single number of the elapsed seconds between timepoints
			None, in which case this information is taken from file
			modification time
		Also creates a list of timepoints
		'''
		### !!! NEEDS UNITTEST
		self.timepoint_list = \
			range(first_timepoint, (self.total_timepoint_num + 1))
		if type(timepoint_spacing) is list:
			self.timepoint_dict = \
				dict(zip(self.timepoint_list, timepoint_spacing))
		elif type(timepoint_spacing) is int or type(timepoint_spacing) is float:
			timepoint_spacing_vector = \
				list(np.array(self.timepoint_list, dtype = float) * 
					timepoint_spacing)
			self.timepoint_dict = \
				dict(zip(self.timepoint_list, timepoint_spacing_vector))
		elif timepoint_spacing is None:
			self.timepoint_dict	= None
		else:
			raise TypeError('timepoint_spacing must be either a list of ' +
				'numbers of same length as the number of timepoints in the ' +
				'experiment, a single integer/float, or None')

	def _find_first_timepoint(self):
		'''
		Finds and writes the time of the first timepoint of this imaging
		phase
		'''
		### !!! NEEDS UNITTEST
		first_timepoint_file = \
			os.path.join(self.phase_output_path, 'first_timepoint_time.txt')
		try:
			with open(first_timepoint_file) as f:
				self.first_timepoint_time = int(f.readline())
		except:
			if self.timepoint_dict is None:
				# if no timepoint dict, find the modification time of
				# the first image captured in this phase
				self.first_timepoint_time = np.inf
				for current_file in os.listdir(self.input_path):
					if current_file.endswith(self.im_file_extension):
						current_time = \
							os.path.getmtime(os.path.join(self.input_path,
								current_file))
						self.first_timepoint_time = \
							np.min([self.first_timepoint_time, current_time])
			else:
				self.first_timepoint_time = \
					self.timepoint_dict[self.timepoint_list[0]]
			# write to text file
			with open(first_timepoint_file, 'w') as f:
				f.write('%d' % self.first_timepoint_time)

	def _run_parameter_tests(self):
		'''
		Runs tests to ensure certain parameters have correct values
		'''
		if self.min_colony_area < 0:
			raise ValueError('min_colony_area must be 0 or more')

	def get_position_colony_data_tracked_df(self, remove_untracked = False):
		'''
		Reads and returns tracked colony properties for the current
		xy position
		If remove_untracked is true, removes any rows with missing
		time_tracking_id (corresponding to colonies that weren't tracked
		because e.g. they are a minor piece of a broken-up colony)
		'''
		pos_tracked_col_prop_df = \
			pd.read_parquet(self.tracked_properties_write_path)
		if remove_untracked:
			pos_tracked_col_prop_df = \
				pos_tracked_col_prop_df[
					pos_tracked_col_prop_df.time_tracking_id.notna()]
		return(pos_tracked_col_prop_df)

	def get_colony_data_tracked_df(self, remove_untracked = False,
								filter_by_phase = True):
		'''
		Reads and returns dataframe of tracked phase colony properties
		output

		If remove_untracked is True,
		removes any rows with missing time_tracking_id (corresponding
		to colonies that weren't tracked because e.g. they are a minor
		piece of a broken-up colony)
		'''
		colony_properties_df_total = \
			pd.read_csv(self.combined_tracked_properties_write_path)
		if filter_by_phase:
			colony_properties_df = colony_properties_df_total[
				colony_properties_df_total.phase_num == self.phase_num]
		else:
			colony_properties_df = colony_properties_df_total
		if remove_untracked and not colony_properties_df.empty:
			colony_properties_df = \
				colony_properties_df[
					colony_properties_df.time_tracking_id.notna()]
		return(colony_properties_df)

	def get_gr_data(self):
		'''
		Reads and returns dataframe of growth rates
		'''
		gr_df = pd.read_csv(self.combined_gr_write_path)
		return(gr_df)

	def get_property_mat_path(self, col_property):
		'''
		Gets path to property matrix
		'''
		write_path = os.path.join(self.phase_col_property_mats_output_folder,
			(col_property + '_property_mat.csv'))
		return(write_path)


class _AnalysisConfigFileProcessor(object):
	'''
	Reads an analysis config csv file and creates a dictionary with an
	AnalysisConfig object for each phase of the experiment; for each
	phase, the 0 position in the list stored in the dictionary is the
	phase analysis config, and the 1 position is the postphase analysis
	config
	'''
	# TODO: Maybe need some safety checks to see that things you'd
	# expect to be the same across all phases (e.g. position numbers)
	# actually are?
	def __init__(self):
		# set default parameter values to be used for every phase; any 
		# of these that are different in the setup file will be 
		# modiefied based on that
		self._default_param_ser = pd.Series({
			'hole_fill_area':np.inf,
			'cleanup':False,
			'max_proportion_exposed_edge':0.75,
			'perform_registration':True,
			'im_file_extension':'tif',
			'minimum_growth_time':1,
			'total_xy_position_num':1,
			'first_timepoint':1,
			'timepoint_label_prefix':'t',
			'position_label_prefix':'xy',
			'main_channel_label':'',
			'main_channel_imagetype':'brightfield',
			'im_format':'individual',
			'extended_display_positions':[1],
			'first_xy_position':1,
			'max_area_pixel_decrease':np.inf,
			'max_area_fold_decrease':2,
			'max_area_fold_increase':6,
			'min_colony_area':10,
			'max_colony_area':np.inf,
			'min_correlation':0.9,
			'min_foldX':0,
			'max_colony_num':1000,
			'fluor_channel_scope_labels':'',
			'fluor_channel_names':'',
			'fluor_channel_thresholds':'',
			'fluor_channel_timepoints':'',
			'parent_phase':''
			})

	def _convert_to_number(self, val_str):
		'''
		Converts val_str to an int or float or logical (in that order) if
		possible
		'''
		# NEED UNITTEST FOR JUST THIS METHOD?
		try:
			output_val = int(val_str)
		except:
			try:
				output_val = float(val_str)
			except:
				if val_str.lower() == 'true':
					output_val = True
				elif val_str.lower() == 'false':
					output_val = False
				else:
					output_val = val_str
		return(output_val)

	def _process_parameter_vals(self, val_str):
		'''
		Returns val_str split by semicolon into list only if semicolon
		is present
		Converts val_str or all elements of resulting list to int if
		possible
		'''
		# NEED UNITTEST FOR JUST THIS METHOD?
		if ';' in val_str:
			split_val_str = val_str.split(';')
			output_val = [self._convert_to_number(val) for val in split_val_str]
		else:
			output_val = self._convert_to_number(val_str)
		return(output_val)

	def _unprocess_parameter_vals(self, val):
		'''
		Joins any lists with semicolons
		'''
		if isinstance(val, list):
			val_str = ';'.join([str(x) for x in val])
		else:
			# don't convert to str here or nan will be written as 'nan' 
			# and not as blank
			val_str = val
		return(val_str)

	def _set_global_vals(self):
		'''
		Check that parameters in config file related to number of
		imaging positions, analysis output, and format in which input
		images are saved apply across all phases, and set them as 
		attributes of self
		'''
		### !!! NEEDS UNITTEST!
		required_global_params = \
			['output_path', 'im_format', 'first_xy_position',
			'total_xy_position_num', 'extended_display_positions']
		global_param_set = set(self._global_param_ser.index)
		if not set(required_global_params).issubset(global_param_set):
			raise ValueError((
				'The following parameters must have PhaseNum set to "all",'
				' or be identical across all phases: {0}\nOf these, {1} '
				'differs among phases.').format(
					', '.join(required_global_params),
					', '.join(list(set.difference(
						set(required_global_params), global_param_set
						)))
					)
				)
		self.output_path = self._global_param_ser.output_path
		self.im_format = self._global_param_ser.im_format
		self.total_xy_position_num = \
			self._global_param_ser.total_xy_position_num
		# set up list of possible xy positions
		self.xy_position_vector = \
			range(self._global_param_ser.first_xy_position,
				(self.total_xy_position_num + 1))
		if not self._global_param_ser.extended_display_positions:
			self.extended_display_positions = []
		elif isinstance(
			self._global_param_ser.extended_display_positions,
			int):
			self.extended_display_positions = [
				self._global_param_ser.extended_display_positions]
		else:
			self.extended_display_positions = \
				self._global_param_ser.extended_display_positions

	def _check_phase_numbers(self, analysis_config_df_prelim):
		'''
		Checks that PhaseNum column contains only 'all' or integers, 
		throws warning about dropping any non-blanks

		If PhaseNum isn't specified at all, sets it to '1' for all rows

		Removes any columns where PhaseNum or Parameter not specified 
		correctly
		'''
		if not 'PhaseNum' in analysis_config_df_prelim.columns:
			analysis_config_df_prelim['PhaseNum'] = str(1)
		phase_num_vals = np.array([
			phase.lower().strip() for phase in 
			analysis_config_df_prelim.PhaseNum
			])
		phase_num_pass_bool = np.array([
			phase.isdigit() or phase=='all' for phase in phase_num_vals
			])
		phase_num_fail_vals = phase_num_vals[~phase_num_pass_bool]
		if any(phase_num_fail_vals!=''):
			drop_phases = \
				list(np.unique(phase_num_fail_vals[phase_num_fail_vals!='']))
			warnings.warn((
				"PhaseNum may only be 'all' or an integer; dropping disallowed"
				" phases {0}"
				).format(str(drop_phases)), UserWarning)
		# only keep rows with allowed phase num
		drop_indices = analysis_config_df_prelim.index[np.logical_or(
			~phase_num_pass_bool, analysis_config_df_prelim.Parameter == ''
			)]
		analysis_config_df_prelim.drop(index = drop_indices, inplace = True)
		return(analysis_config_df_prelim)

	def _check_req_completeness(self, setup_ser, required_fields):
		'''
		Checks whether all parameters in setup_ser are present in 
		required_fields; if not, raises error
		'''
		missing_fields = \
			set.difference(set(required_fields), set(setup_ser.index))
		if len(missing_fields) > 0:
			raise ValueError((
				'Missing required fields {0} in PhaseNum {1}; '
				'if your experiment has multiple phases, check that '
				'you have specified every parameter for every phase (either '
				'individually or my marking "all" under PhaseNum)'
				).format(str(missing_fields), str(setup_ser.name))
				)

	def _check_extra_params(self, setup_ser, required_fields):
		'''
		Checks whether any parameters are in setup_ser that aren't present in 
		required_fields; if there are, raise warning
		'''
		extra_fields = \
			set.difference(set(setup_ser.index), set(required_fields))
		if len(extra_fields) > 0:
			warnings.warn(
				('Unused parameters: {0}').format(str(extra_fields)),
				UserWarning
				)
			setup_ser.drop(list(extra_fields), inplace = True)
		return(setup_ser)

	def _create_phase_conf_ser(
		self,
		template_setup_ser,
		current_setup_ser,
		required_fields
		):
		# NEED UNITTEST!!!
		'''
		Generates a pandas series, phase_conf_ser, that inherits
		parameters from template_setup_ser unless they're also specified in
		current_setup_ser, in which case the parameters in
		current_setup_ser are used
		'''
		# take all possible fields from current_setup_ser, get missing
		# ones from template_setup_ser
		reqd_template_fields = \
			set.difference(set(required_fields), set(current_setup_ser.index))
		# create combined series from template fields that are missing 
		# in current_setup_ser, and current_setup_ser
		template_subset_ser_to_use = \
			template_setup_ser[
				list(
					set.intersection(
						reqd_template_fields,
						set(template_setup_ser.index)
						)
					)
				]
		phase_conf_ser = pd.concat(
			[template_subset_ser_to_use, current_setup_ser]
			)
		return(phase_conf_ser)

	def _organize_config_df(self, analysis_config_df_prelim):
		'''
		Creates self.analysis_config_df with PhaseNum as columns, and 
		self._global_param_ser that contains values for global params
		'''
		# pivot analysis_config_df to have phases as columns
		analysis_config_df_pivot = analysis_config_df_prelim.pivot(
			index = 'Parameter', columns = 'PhaseNum', values = 'Value'
			)
		if 'all' in analysis_config_df_pivot.columns:
			# initialize global param series
			global_param_ser_part = analysis_config_df_pivot['all'].dropna()
			# drop 'all' column from analysis_config_df_pivot
			analysis_config_df_indiv = \
				analysis_config_df_pivot.drop(
					columns = ['all']
					).dropna(axis = 0, how = 'all')
		else:
			global_param_ser_part = pd.Series(name = 'all', dtype = object)
			analysis_config_df_indiv = \
				analysis_config_df_pivot
		# convert column names of analysis_config_df to int and use to 
		# specify phases
		analysis_config_df_indiv.columns = analysis_config_df_indiv.columns.astype(int)
		# define phases
		self.phases = analysis_config_df_indiv.columns.copy().to_list()
		# check that parameters defined as 'all' and parameters defined 
		# by individual phases are mutually exclusive
		indiv_phase_param_set = set(analysis_config_df_indiv.index)
		global_param_set = set(global_param_ser_part.index)
		double_defined_params = \
			set.intersection(indiv_phase_param_set, global_param_set)
		if len(double_defined_params) > 0:
			raise ValueError((
				'Parameters may be defined either with PhaseNum set to "all" '
				'or set to individual phase number integers; the following '
				'parameters were defined with both: \n{0}'
				).format(str(double_defined_params)))
		# Add all parameters with 
		# identical values across phases to global_param_ser_part, and 
		# remove those parameters from analysis_config_df_indiv
		global_param_row_bool_ser = analysis_config_df_indiv.eq(
			analysis_config_df_indiv.iloc[:, 0], axis=0
			).all(axis=1)
		if any(global_param_row_bool_ser):
			new_global_params = \
				global_param_row_bool_ser.index[global_param_row_bool_ser]
			global_param_ser_part = global_param_ser_part.append(
				analysis_config_df_indiv.loc[
					new_global_params, analysis_config_df_indiv.columns[0]
					]
				)
			analysis_config_df_indiv.drop(index = new_global_params, inplace = True)
		# create self._global_param_ser by using default values for any 
		# parameters still missing from both analysis_config_df_indiv 
		# and from global_param_ser_part
		specified_indiv_phase_default_params = list(set.intersection(
			set(analysis_config_df_indiv.index),
			set(self._default_param_ser.index)
			))
		self._global_param_ser = self._create_phase_conf_ser(
			self._default_param_ser.drop(specified_indiv_phase_default_params),
			global_param_ser_part,
			required_fields_general
			)
		self._global_param_ser = self._check_extra_params(
			self._global_param_ser, required_fields_general
			)
		# create a subset of default parameters for phase-specific
		# params only
		default_param_ser_indiv = \
			self._default_param_ser[specified_indiv_phase_default_params]
		# for each phase, fill in missing values with defaults
		analysis_config_dict = dict()
		# if no parent_phase in any phases, skip over that part
		if 'parent_phase' in self._global_param_ser.index:
			if self._global_param_ser.parent_phase == '':
				self._no_postphase = True
			elif len(self.analysis_config_df.columns) > 1:
				raise ValueError(
					'If parent_phase is specified for all phases ' +
						'simultaneously, it must be left blank')
		else:
			self._no_postphase = False
		for phase in self.phases:
			# get only the parameters specified for the current phase
			curr_phase_vals_part = analysis_config_df_indiv[phase].dropna()
			if self._no_postphase or \
				(('parent_phase' not in curr_phase_vals_part.index) &
					(self._default_param_ser.parent_phase in ['', phase])) or \
				curr_phase_vals_part.parent_phase in ['', phase]:
				curr_req_fields = required_fields_general
			else:
				curr_req_fields = required_fields_minimal
			curr_phase_vals_full = self._create_phase_conf_ser(
				default_param_ser_indiv,
				curr_phase_vals_part,
				curr_req_fields
				)
			curr_phase_vals_full = \
				self._check_extra_params(curr_phase_vals_full, curr_req_fields)
			analysis_config_dict[phase] = curr_phase_vals_full
		self.analysis_config_df = pd.DataFrame(analysis_config_dict)

	def _create_fluor_channel_df(self, phase_conf_ser, phase_num):
		'''
		Creates a dataframe from phase_conf_ser with info on every
		fluorescent channel imaged
		'''
		### !!! NEED UNITTEST?
		# create empty df if every fluor property is an empty string
		list_of_fluor_properties = [phase_conf_ser.fluor_channel_scope_labels,
			phase_conf_ser.fluor_channel_names,
			phase_conf_ser.fluor_channel_thresholds,
			phase_conf_ser.fluor_channel_timepoints]
		if all([x == '' for x in list_of_fluor_properties]):
			fluor_channel_df = pd.DataFrame(columns =
				['fluor_channel_label', 'fluor_channel_column_name',
					'fluor_threshold', 'fluor_timepoint'])
		else:
			# create df with a row for every channel
			# (use np.size here, not len function, to get accurate
			# lengths for single-string fluor_channel_scope_labels)
			channel_num = np.size(phase_conf_ser.fluor_channel_scope_labels)
			fluor_channel_df = \
				pd.DataFrame({
					'fluor_channel_label':
						phase_conf_ser.fluor_channel_scope_labels,
					'fluor_channel_column_name':
						phase_conf_ser.fluor_channel_names,
					'fluor_threshold':
						phase_conf_ser.fluor_channel_thresholds,
					'fluor_timepoint':
						phase_conf_ser.fluor_channel_timepoints},
					index = np.arange(0, channel_num))
			# raise error if only some fluor properties are empty strings
			mutually_required_fluor_properties = [
				'fluor_channel_column_name',
				'fluor_threshold',
				'fluor_timepoint']
			for prop in mutually_required_fluor_properties:
				if '' in fluor_channel_df[prop]:
					raise ValueError(
						prop + 
						' is not set for one of the channels in phase ' +
						str(phase_num) +
						'; these values must either all be left blank, or '
						'all filled')
			# raise error if any non-unique values in columns
			unique_properties = [
				'fluor_channel_label',
				'fluor_channel_column_name']
			for prop in unique_properties:
				if not fluor_channel_df[prop].is_unique:
					raise ValueError(
						'Non-unique values identified in ' + prop +
						'for phase ' + str(phase_num) + ': ' +
						str(fluor_channel_df[prop]))
		return(fluor_channel_df)

	def _create_analysis_config(self, phase_num, phase_conf_ser):
		'''
		Creates AnalysisConfig object based on phase_conf_ser, the
		series corresponding to the Value column of the subset of
		self.analysis_config_df that applies to the current phase
		'''
		### NEED UNITTEST FOR JUST THIS METHOD?
		self._check_req_completeness(phase_conf_ser, required_fields_general)
		fluor_channel_df = \
			self._create_fluor_channel_df(phase_conf_ser, phase_num)
		# if timepoint spacing tab is empty, set timepoint_spacing to
		# None (i.e. get info from files)
		if phase_conf_ser.timepoint_spacing == '':
			timepoint_spacing = None
		else:
			timepoint_spacing = phase_conf_ser.timepoint_spacing
		# create AnalysisConfig object
		current_analysis_config = AnalysisConfig(
			phase_num,
			phase_conf_ser.hole_fill_area,
			phase_conf_ser.cleanup,
			phase_conf_ser.perform_registration,
			phase_conf_ser.max_proportion_exposed_edge,
			phase_conf_ser.input_path,
			phase_conf_ser.output_path,
			phase_conf_ser.im_file_extension,
			phase_conf_ser.label_order_list,
			phase_conf_ser.total_xy_position_num,
			phase_conf_ser.first_timepoint,
			phase_conf_ser.total_timepoint_num,
			timepoint_spacing,
			phase_conf_ser.timepoint_label_prefix,
			phase_conf_ser.position_label_prefix,
			phase_conf_ser.main_channel_label,
			phase_conf_ser.main_channel_imagetype,
			fluor_channel_df,
			phase_conf_ser.im_format,
			self.extended_display_positions,
			self.xy_position_vector,
			phase_conf_ser.minimum_growth_time,
			phase_conf_ser.growth_window_timepoints,
			phase_conf_ser.max_area_pixel_decrease,
			phase_conf_ser.max_area_fold_decrease,
			phase_conf_ser.max_area_fold_increase,
			phase_conf_ser.min_colony_area,
			phase_conf_ser.max_colony_area,
			phase_conf_ser.min_correlation,
			phase_conf_ser.min_foldX,
			phase_conf_ser.min_neighbor_dist,
			phase_conf_ser.max_colony_num)
		return(current_analysis_config)

	def _create_postphase_analysis_config(self, phase_num, phase_conf_ser):
		'''
		Creates MinimalAnalysisConfig object based on phase_conf_ser, 
		the series corresponding to the Value column of the subset of
		self.analysis_config_df that applies to the current phase
		'''
		### NEED UNITTEST FOR JUST THIS METHOD?
		self._check_req_completeness(phase_conf_ser, required_fields_minimal)
		fluor_channel_df = \
			self._create_fluor_channel_df(phase_conf_ser, phase_num)
		# create MinimalAnalysisConfig object
		postphase_analysis_config = MinimalAnalysisConfig(
			phase_num,
			phase_conf_ser.input_path,
			phase_conf_ser.output_path,
			phase_conf_ser.im_file_extension,
			phase_conf_ser.label_order_list,
			phase_conf_ser.total_xy_position_num,
			phase_conf_ser.position_label_prefix,
			fluor_channel_df,
			phase_conf_ser.im_format,
			self.extended_display_positions,
			phase_conf_ser.timepoint_label_prefix,
			self.xy_position_vector
			)
		return(postphase_analysis_config)

	def _create_analysis_config_df(self):
		'''
		Loops through phases and creates a pandas df of AnalysisConfig
		objects
		'''
		# NEED UNITTEST FOR JUST THIS METHOD?
		# create a pandas df for storing AnalysisConfig objects
		analysis_config_obj_df = \
			pd.DataFrame({'analysis_config': None,
				'postphase_analysis_config': None}, index = self.phases)
		# create a phase setup series containing the parameters that
		# apply to all phases
		for phase_num in self.phases:
			# get phase setup series containing the parameters that
			# apply only to the current phase_num
			# safe to simply concatenate parameter series here because 
			# all defaults should be filled in and all global 
			# parameters separated out
			current_phase_setup = pd.concat([
				self._global_param_ser,
				self.analysis_config_df[phase_num].dropna()
				])
			current_phase_setup.name = phase_num
			# set where to store object
			if self._no_postphase or current_phase_setup.parent_phase in \
				['',phase_num]:
				storage_phase = phase_num
				config_type = 'analysis_config'
				# create AnalysisConfig object from current phase setup ser
				current_analysis_config = \
					self._create_analysis_config(
						phase_num,
						current_phase_setup
						)
			elif not (
				self.analysis_config_df.at[
					'parent_phase', current_phase_setup.parent_phase
					]
				in ['', current_phase_setup.parent_phase]
				):
				# check that parent_phase doesn't have its own 
				# parent_phase
				raise ValueError(
					(
						'Phase {0}\'s parent phase listed as Phase {1}, but '
						'that has its own parent phase.'
						).format(
							str(phase_num),str(current_phase_setup.parent_phase)
							)
					)
			else:
				# treat as postphase
				storage_phase = current_phase_setup.parent_phase
				config_type = 'postphase_analysis_config'
				# create AnalysisConfig object from current phase setup ser
				current_analysis_config = \
					self._create_postphase_analysis_config(
						phase_num,
						current_phase_setup
						)
			# store AnalysisConfig object in pandas df
			analysis_config_obj_df.at[storage_phase, config_type] = \
				current_analysis_config
		# in case any phase rows are empty (because those phases are
		# child phases of other phases), remove rows with all None
		analysis_config_obj_df.dropna(0, how = 'all', inplace = True)
		return(analysis_config_obj_df)

	def write_setup_file(self, setup_file_out_path):
		'''
		Writes csv file containing all parameters from 
		self._global_param_ser and self.analysis_config_df to
		setup_file_out_path
		'''
		# make dataframe for global params
		global_param_df = \
			pd.DataFrame(self._global_param_ser, columns = ['Value'])
		# convert index to 'Parameter' column
		global_param_df = global_param_df.reset_index().rename(
			columns = {'index':'Parameter'}
			)
		# add PhaseNum column
		global_param_df['PhaseNum'] = 'all'
		# make dataframe for phase-specific params
		phasewise_param_df = self.analysis_config_df.melt(
			value_vars = self.phases,
			var_name = 'PhaseNum',
			value_name = 'Value',
			ignore_index = False
			)
		# drop rows where Value is NA
		phasewise_param_df.dropna(subset = ['Value'], inplace = True)
		# convert index to 'Parameter' column
		phasewise_param_df = phasewise_param_df.reset_index().rename(
			columns = {'index':'Parameter'}
			)
		# combine global and phasewise dfs
		combined_df_prelim = pd.concat([global_param_df, phasewise_param_df])
		# add explanation column and reorder columns
		combined_df = pd.merge(param_description_df,combined_df_prelim)
		combined_df = \
			combined_df[['Parameter', 'Value', 'PhaseNum', 'Explanation']]
		# convert lists in Parameter to semicolon-joined strings
		combined_df.Value = combined_df.Value.map(
			self._unprocess_parameter_vals
			)
		combined_df.to_csv(setup_file_out_path, index = False)

	def process_analysis_config_file(self, analysis_config_path):
		'''
		Reads csv file in analysis_config_path and creates pandas df of
		AnalysisConfig objects for each phase
		'''
		# read in config file
		# convert strings to int where possible, and convert values
		# separated by semicolon to lists
		analysis_config_df_prelim = pd.read_csv(
			analysis_config_path,
			dtype = {'PhaseNum': str},
			converters =
				{'Value': self._process_parameter_vals},
			na_filter = False
			)
		# check file format
		if not {'Parameter','Value'}.issubset(
			set(analysis_config_df_prelim.columns)
			):
			raise IndexError(
				'Could not find columns Parameter, Value, and PhaseNum in your'
				' setup file, ' + analysis_config_path + '; the most common '
				'cause of this is that your file is not correctly saved in '
				'comma-separated mode. You can check this by opening the file '
				'in a text editor (e.g. wordpad). If unable to resolve this '
				'issue, create a new setup file from scratch using PIE\'s '
				'setup wizard'
				)
		# add PhaseNum if it's not specified
		# drop rows where PhaseNum or Parameter is not specified correctly
		analysis_config_df_prelim_phase_filt = \
			self._check_phase_numbers(analysis_config_df_prelim)
		# set up global and phase-specific parameter dfs/series
		self._organize_config_df(analysis_config_df_prelim_phase_filt)
		# check that phase for global parameters correctly specified,
		# and set them as attributes
		self._set_global_vals()
		# create df of analysis config objects
		analysis_config_obj_df = self._create_analysis_config_df()
		return(analysis_config_obj_df)

def process_setup_file(analysis_config_path):
	'''
	Processed the experimental setup file in analysis_config_path and
	creates pandas df of AnalysisConfig objects for each phase
	If there's no setup file saved in output path, copies file in
	analysis_config_path to output_path/setup_file.csv
	'''
	analysis_config_file_processor = _AnalysisConfigFileProcessor()
	analysis_config_obj_df = \
		analysis_config_file_processor.process_analysis_config_file(
			analysis_config_path)
	# save setup file in output path
	setup_file_out_path = \
		os.path.join(analysis_config_file_processor.output_path,
			'setup_file.csv')
	if not os.path.exists(setup_file_out_path):
		analysis_config_file_processor.write_setup_file(setup_file_out_path)
	return(analysis_config_obj_df)

def check_passed_config(analysis_config_obj_df, analysis_config_file):
	'''
	Check that only analysis_config_obj_df or analysis_config_file is
	passed, and get analysis_config_obj_df
	'''
	if (analysis_config_obj_df is None) == (analysis_config_file is None):
		raise ValueError(
			'Must supply EITHER analysis_config_obj_df OR ' +
			'analysis_config_file argument')
	if analysis_config_obj_df is None:
		analysis_config_obj_df = process_setup_file(analysis_config_file)
	return(analysis_config_obj_df)



