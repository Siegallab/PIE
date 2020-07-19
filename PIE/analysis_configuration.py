#!/usr/bin/python

'''
Tracks colonies through time in a single imaging field
'''

import cv2
import numpy as np
import os
import pandas as pd
from random import sample

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

class AnalysisConfig(object):
	'''
	Handles experimental configuration details
	'''
	def __init__(self, phase, hole_fill_area, cleanup,
		max_proportion_exposed_edge, input_path, output_path, im_file_extension,
		label_order_list, total_xy_position_num, first_timepoint,
		total_timepoint_num, timepoint_spacing, timepoint_label_prefix,
		position_label_prefix, main_channel_label, main_channel_imagetype,
		fluor_channel_df, im_format, chosen_for_extended_display_list,
		first_xy_position, settle_frames, minimum_growth_time,
		growth_window_timepoints, max_area_pixel_decrease,
		max_area_fold_decrease, max_area_fold_increase, min_colony_area,
		max_colony_area, min_correlation, min_foldX, min_neighbor_dist):
		'''
		Reads setup_file and creates analysis configuration
		'''
		# specify phase
		self.phase = phase
		# max xy position label and timepoint number
		self.total_xy_position_num = int(total_xy_position_num)
		self.total_timepoint_num = int(total_timepoint_num)
		# specify image analysis parameters
		self.hole_fill_area = float(hole_fill_area)
		self.cleanup = bool(cleanup)
		self.max_proportion_exposed_edge = float(max_proportion_exposed_edge)
		# specify growth rate analysis parameters
		self.settle_frames = int(settle_frames)
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
		# path of input images
		self.input_path = input_path
		# path of output image folder
		self.output_path = output_path
		# file extension of input images
		self.im_file_extension = im_file_extension
		# save order in which time, position, channel labels are listed,
		# after checking that it contains the necessary info
		if set(label_order_list) == set(['timepoint', 'channel', 'position']):
			self.label_order_list = label_order_list
		else:
			raise ValueError(
				'Label order list must consist of timepoint, channel, and ' +
				'position, even if not all these are used')
		# set up folder to save output of positionwise phase results
		self._create_phase_output()
		# set up dictionary of timepoint times
		self._set_up_timevector(timepoint_spacing, first_timepoint)
		# set up list of possible xy positions
		self.xy_position_vector = \
			range(first_xy_position, (self.total_xy_position_num + 1))
		# labels used for timepoint number and xy position
		self.timepoint_label_prefix = timepoint_label_prefix
		self.position_label_prefix = position_label_prefix
		# find time of first existing file
		self._find_first_timepoint()
		# specify type of image (brightfield or phase_contrast) is in
		# the main channel
		self.main_channel_imagetype = main_channel_imagetype
		# set up channel labels
		self.main_channel_label = main_channel_label
		self.fluor_channel_df = fluor_channel_df
			# column names: fluor_channel_label,
			# fluor_channel_column_name, fluor_threshold
		# set up image retriever depending on im_format
		if im_format == 'individual':
			self.image_retriever = _IndivImageRetriever()
		else:
			raise ValueError('image format ' + im_format + ' not recognized')
		self.chosen_for_extended_display_list = chosen_for_extended_display_list
		self._run_parameter_tests()

	def _create_phase_output(self):
		'''
		Creates folder for results of colony properties across phases,
		as well as within-phase results, if they don't already exist
		'''
		# NEED UNITTEST FOR JUST THIS METHOD?
		self.phase_output_path = os.path.join(self.output_path,
			('phase_' + self.phase))
		self.phase_col_properties_output_folder = \
			os.path.join(self.phase_output_path, 'positionwise_colony_properties')
		if not os.path.exists(self.phase_col_properties_output_folder):
			os.makedirs(self.phase_col_properties_output_folder)
		if not os.path.exists(self.phase_output_path):
			os.makedirs(self.phase_output_path)
		# create filename for tracked colony properties output file for
		# current position
		self.phase_tracked_properties_write_path = \
			os.path.join(self.phase_output_path,
				'col_props_with_tracking_pos.csv')
		# create filename for growth rate output file for current
		# position
		self.phase_gr_write_path = \
			os.path.join(self.phase_output_path, 'growth_rates.csv')

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
		Finds and writes the time of the first timepoint of this imaging phase
		'''
		### !!! NEEDS UNITTEST
		first_timepoint_file = \
			os.path.join(self.phase_output_path, 'first_timepoint_time.txt')
		if os.path.exists(first_timepoint_file):
			with open(first_timepoint_file) as f:
				self.first_timepoint = int(f.readline())
		else:
			if self.timepoint_dict is None:
				# if no timepoint dict, find the modification time of
				# the first image captured in this phase
				self.first_timepoint = np.inf
				for current_file in os.listdir(self.input_path):
					if current_file.endswith(self.im_file_extension):
						current_time = \
							os.path.getmtime(os.path.join(self.input_path,
								current_file))
						self.first_timepoint = \
							np.min([self.first_timepoint, current_time])
			else:
				self.first_timepoint = \
					self.timepoint_dict[self.timepoint_list[0]]
			# write to text file
			with open(first_timepoint_file, 'w') as f:
  				f.write('%d' % self.first_timepoint)

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

	def _run_parameter_tests(self):
		'''
		Runs tests to ensure certain parameters have correct values
		'''
		if self.min_colony_area < 0:
			raise ValueError('min_colony_area must be 0 or more')

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

	def get_colony_data_tracked_df(self):
		'''
		Reads and returns dataframe of tracked phase colony properties
		output
		Removes any rows with missing unique_tracking_id (corresponding
		to colonies that weren't tracked because e.g. they are a minor
		piece of a broken-up colony)
		'''
		colony_properties_df = \
			pd.read_csv(self.phase_tracked_properties_write_path,
				index_col = 0)
		colony_properties_tracked_only = \
			colony_properties_df[
				colony_properties_df.unique_tracking_id.notna()]
		return(colony_properties_tracked_only)

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
		if image is None:
			image_time = None
		else:
			# if timepoint dict exists, get time value from there;
			# otherwise, get it from the file modification date
			if self.timepoint_dict:
				image_time = self.timepoint_dict[timepoint]
			else:
				image_time = os.path.getmtime(im_filepath)
			# update minimum image_time of phase
		return(image, im_label, image_time)

	def set_xy_position(self, xy_position_idx):
		'''
		Sets the xy position to be used by the analysis config
		'''
		### !!! NEEDS UNITTEST
		if xy_position_idx not in self.xy_position_vector:
			raise IndexError('Unexpected xy position index ' + xy_position_idx +
				' in phase ' + self.phase)
		# current position being imaged
		self.xy_position_idx = xy_position_idx
		# determine wether non-essential info (threshold plot outputs,
		# boundary images, etc) need to be saved for this experiment
		self.save_extra_info = \
			xy_position_idx in self.chosen_for_extended_display_list

class AnalysisConfigFileProcessor(object):
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
				elif val_str == '':
					output_val = np.nan
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

	def _define_phases(self):
		'''
		Identifies the phases in the experiment
		Phases are treated as lower-case strings
		'''
		# NEED UNITTEST FOR JUST THIS METHOD?
		unique_phases = self.analysis_config_df.Phase.astype(str).unique()
		# only keep phases that are not "all"
		self.phases = [phase.lower() for phase in unique_phases if
			phase.lower() != 'all']

	def _create_analysis_config(self, phase, phase_conf_ser):
		'''
		Creates AnalysisConfig object based on phase_conf_ser, the
		series corresponding to the Value column of the subset of
		self.analysis_config_df that applies to the current phase
		'''
		# NEED UNITTEST FOR JUST THIS METHOD?
		fluor_col_names = ['fluor_channel_label', 'fluor_channel_column_name',
			'fluor_threshold']
		list_of_fluor_properties = [phase_conf_ser.fluor_channel_scope_labels,
			phase_conf_ser.fluor_channel_names,
			phase_conf_ser.fluor_channel_thresholds]
		if all(np.isnan(list_of_fluor_properties)):
			fluor_channel_df = pd.DataFrame(columns = fluor_col_names)
		elif any(np.isnan(list_of_fluor_properties)):
			raise ValueError(
				'fluor_channel_label, fluor_channel_column_nam, or ' +
				'fluor_threshold is not set for phase ' + phase +
				'; these values must either all be left blank, or all filled')
		else:
			fluor_channel_df = \
				pd.DataFrame(list(zip(*list_of_fluor_properties)),
				columns = fluor_col_names)
		# if timepoint spacing tab is empty, set timepoint_spacing to
		# None (i.e. get info from files)
		if np.isnan(phase_conf_ser.timepoint_spacing):
			timepoint_spacing = None
		else:
			timepoint_spacing = phase_conf_ser.timepoint_spacing
		# create AnalysisConfig object
		current_analysis_config = AnalysisConfig(
			phase,
			phase_conf_ser.hole_fill_area,
			phase_conf_ser.cleanup,
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
			self.chosen_for_extended_display_list,
			phase_conf_ser.first_xy_position,
			phase_conf_ser.settle_frames,
			phase_conf_ser.minimum_growth_time,
			phase_conf_ser.growth_window_timepoints,
			phase_conf_ser.max_area_pixel_decrease,
			phase_conf_ser.max_area_fold_decrease,
			phase_conf_ser.max_area_fold_increase,
			phase_conf_ser.min_colony_area,
			phase_conf_ser.max_colony_area,
			phase_conf_ser.min_correlation,
			phase_conf_ser.min_foldX,
			phase_conf_ser.min_neighbor_dist)
		return(current_analysis_config)

	def _get_phase_data(self, phase):
		'''
		Extracts data corresponding to phase in Phase column from
		self.analysis_config_df into a pandas series, changing index
		to Parameter column
		phase is a lower-case string
		'''
		# NEED UNITTEST FOR JUST THIS METHOD?
		# extract data
		current_phase_data = \
			self.analysis_config_df[self.analysis_config_df.Phase.str.lower() == phase]
		# check that each parameter is specified only once in this phase
		if not current_phase_data.Parameter.is_unique:
			raise ValueError('There is a Parameter value listed in phase ' +
				phase + ' of the setup file that is not unique.')
		# set index to parameter and extract series corresponding to values
		current_phase_data.index = current_phase_data.Parameter
		current_phase_series = current_phase_data.Value
		return(current_phase_series)

	def _create_phase_conf_ser(self, parent_setup_ser, current_setup_ser,
		check_req_completeness = True):
		# NEED UNITTEST FOR JUST THIS METHOD?
		'''
		Generates a pandas series, phase_conf_ser, that inherits
		parameters from parent_setup_df unless they're also specified in
		current_setup_df, in which case the parameters in
		current_setup_df are used
		'''
		# list fields that must be in the output series
		required_fields = \
			['fluor_channel_scope_labels', 'fluor_channel_names',
			'fluor_channel_thresholds', 'timepoint_spacing', 'hole_fill_area',
			'cleanup', 'max_proportion_exposed_edge', 'input_path',
			'output_path', 'im_file_extension', 'label_order_list',
			'total_xy_position_num', 'first_timepoint', 'total_timepoint_num',
			'timepoint_label_prefix', 'position_label_prefix',
			'main_channel_label', 'main_channel_imagetype', 'im_format',
			'parent_phase', 'first_xy_position', 'settle_frames',
			'minimum_growth_time', 'growth_window_timepoints']
		# take all possible fields from current_setup_ser, get missing
		# ones from parent_setup_ser
		reqd_parents_fields = \
			set.difference(set(required_fields), set(current_setup_ser.index))
		# check that all required fields are found in parent series
		missing_fields = \
			set.difference(reqd_parents_fields, set(parent_setup_ser.index))
		if check_req_completeness & len(missing_fields) > 0:
			# TODO: There HAS to be a better way to provide helpful error data here
			print('Missing fields: ')
			print(missing_fields)
			print(parent_setup_ser)
			print(current_setup_ser)
			raise IndexError('Missing required fields in one of the previous phase setup series')
		parent_subset_ser_to_use = parent_setup_ser[list(reqd_parents_fields)]
		phase_conf_ser = pd.concat([parent_subset_ser_to_use, current_setup_ser])
		return(phase_conf_ser)

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
		global_parent_setup = self._get_phase_data('all')
		for phase in self.phases:
			# get phase setup series containing the parameters that
			# apply only to the current phase
			current_phase_setup = self._get_phase_data(phase)
			if not np.isnan(current_phase_setup.parent_phase):
				# if the current phase has a 'parent' phase, combine
				# that parent phase with the global parent setup to
				# use as the joint parent phase for the phase
				parent_setup = \
					self._get_phase_data(current_phase_setup.parent_phase)
				combined_parent_setup = \
					self._create_phase_conf_ser(global_parent_setup, parent_setup)
				# set where to store object
				storage_phase = current_phase_setup.parent_phase
				config_type = 'postphase_analysis_config'
			else:
				# if current phase has no parent, just use global setup
				# parent series (i.e. where phase is listed as 'all')
				combined_parent_setup = global_parent_setup
				# set where to store object
				storage_phase = phase
				config_type = 'analysis_config'
			# create setup series based on parent phase(s) and current
			# phase
			current_phase_combined_setup = \
				self._create_phase_conf_ser(combined_parent_setup,
					current_phase_setup)
			# create AnalysisConfig object from current phase setup ser
			current_analysis_config = self._create_analysis_config(phase,
				current_phase_combined_setup)
			# store AnalysisConfig object in pandas df
			analysis_config_obj_df.at[storage_phase, config_type] = \
				current_analysis_config
		return(analysis_config_obj_df)

	def process_analysis_config_file(self, analysis_config_path):
		'''
		Reads csv file in analysis_config_path and creates pandas df of
		AnalysisConfig objects for each phase
		'''
		# read in config file
		# convert strings to int where possible, and convert values
		# separated by semicolon to lists
		self.analysis_config_df = pd.read_csv(analysis_config_path,
			converters =
				{'Value': self._process_parameter_vals})
		self.analysis_config_df.dropna(how="all", inplace=True)
		# get list of elements chosen for extended display
		self.chosen_for_extended_display_list = \
			self.analysis_config_df[
				(self.analysis_config_df['Parameter'] == 
					'extended_display_positions') &
					(self.analysis_config_df['Phase'].str.lower() ==
						'all')].Value.iloc[0]
		if not self.chosen_for_extended_display_list:
			self.chosen_for_extended_display_list = []
		# identify phases
		self._define_phases()
		# create df of analysis config objects
		analysis_config_obj_df = self._create_analysis_config_df()
		return(analysis_config_obj_df)

def set_up_analysis_config(analysis_config_file):
	'''
	Creates a dataframe of phases containing AnalysisConfig objects for
	each phase
	'''
	analysis_config_file_processor = AnalysisConfigFileProcessor()
	analysis_config_obj_df = \
		analysis_config_file_processor.process_analysis_config_file(
			analysis_config_file)
	return(analysis_config_obj_df)