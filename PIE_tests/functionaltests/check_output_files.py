#!/usr/bin/python

import numpy as np
import os
import pandas as pd
import unittest
from PIE import analysis_configuration
from pandas.testing import assert_frame_equal

def make_setup_filepath_standin(config_file_path, new_output_path):
	'''
	Returns StringIO object standing for setup file in config_file_path,
	but with output_path parameter changed to new_output_path
	'''
	os.mkdir(new_output_path)
	analysis_config_file_standin = os.path.join(new_output_path, 'temp_setup.csv')
	analysis_config_df = pd.read_csv(config_file_path)
	# set output path
	analysis_config_df.Value[analysis_config_df.Parameter == 'output_path'] = \
		new_output_path
	analysis_config_df.to_csv(analysis_config_file_standin,
		index = False)
	return(analysis_config_file_standin)

class OutputChecker(unittest.TestCase):
	'''
	Check that all necessary directories and image analysis output files
	expected for a given setup file are created after image analysis run
	If single_pos is an integer, only look for analysis output of
	running PIE.track_colonies_single_pos on that xy_pos_idx
	'''

	def _read_config(self, config_filepath, expected_output_path):
		'''
		Read config file and save important parameters
		'''
		analysis_config_file_processor = \
			analysis_configuration._AnalysisConfigFileProcessor()
		self.analysis_config_obj_df = \
			analysis_config_file_processor.process_analysis_config_file(
				config_filepath)
		self.phases = self.analysis_config_obj_df.index.to_numpy()
		self.output_path = analysis_config_file_processor.output_path
		self.expected_output_path = expected_output_path
		self.positionwise_colony_prop_dir = \
			os.path.join(self.output_path,
				'positionwise_colony_properties')
		self.expected_positionwise_colony_prop_dir = \
			os.path.join(self.expected_output_path,
				'positionwise_colony_properties')
		self.total_xy_position_num = \
			analysis_config_file_processor.total_xy_position_num
		self.xy_position_vector = \
			analysis_config_file_processor.xy_position_vector
		self.extended_display_positions = \
			analysis_config_file_processor.extended_display_positions
		# add phase-specific dirs
		self.analysis_config_obj_df['phase_dir'] = \
			[os.path.join(self.output_path,
				('phase_' + str(p))) for
				p in self.analysis_config_obj_df.index]
		self.analysis_config_obj_df['expected_phase_dir'] = \
			[os.path.join(self.expected_output_path,
				('phase_' + str(p))) for
				p in self.analysis_config_obj_df.index]
		# add threshold info dfs
		self.analysis_config_obj_df['threshold_info_df'] = None
		for phase in self.phases:
			current_threshold_info_file = \
				os.path.join(
					self.analysis_config_obj_df.at[phase, 'phase_dir'],
					'threshold_plots', 'threshold_info.csv'
					)
			try:
				threshold_info_df = \
					pd.read_csv(
						current_threshold_info_file,
						header = None,
						index_col = 0)
			except FileNotFoundError:
				threshold_info_df = pd.DataFrame()
			self.analysis_config_obj_df.at[phase, 'threshold_info_df'] = \
				threshold_info_df
	
	def _read_in_df(self, df_path):
		'''
		Returns dataframe at df_path, which can be either parquet or csv
		'''
		_, ext = os.path.splitext(df_path)
		if ext == '.csv':
			df = pd.read_csv(df_path)
		elif ext == '.parquet':
			df = pd.read_parquet(df_path)
		else:
			raise ValueError(
				'Expecting parquet or csv file, got extension ' + ext)
		return(df)

	def _compare_dataframes(self, expected_df_path, test_df_path):
		'''
		Reads and compares dataframe in expected_df_path to test_df_path
		'''
		expected_df = self._read_in_df(expected_df_path)
		test_df = self._read_in_df(test_df_path)
		assert_frame_equal(expected_df, test_df, check_like = True)

	def _check_directories(self):
		'''
		Check that subdirectories for each phase exist
		'''
		self.assertTrue(os.path.isdir(self.output_path))
		self.assertTrue(self.positionwise_colony_prop_dir)
#		general_output_directories = [
#			'positionwise_colony_property_matrices',
#			'jpgGRimages',
#			'colony_masks']
#		chosen_pos_only_output_directories = [
#			'threshold_plots',
#			'colony_center_overlays',
#			'boundary_ims']
		for phase_dir in self.analysis_config_obj_df['phase_dir']:
			# check that directory for current path exists
			self.assertTrue(os.path.isdir(phase_dir))
#			# check that subdirectories exist
#			expected_phasedir_list = \
#				general_output_directories + \
#				chosen_pos_only_output_directories
#			for d in expected_phasedir_list:
#				print(d)
#				self.assertTrue(os.path.isdir(os.path.join(phase_dir, d)))

	def _check_first_tp(self):
		'''
		Check that correct time is saved into first_timepoint.txt file
		for every phase
		'''
		for phase in self.phases:
			analysis_config = \
				self.analysis_config_obj_df.at[phase, 'analysis_config']
			phase_dir = self.analysis_config_obj_df.at[phase, 'phase_dir']
			first_timepoint_file = \
				os.path.join(phase_dir, 'first_timepoint_time.txt')
			with open(first_timepoint_file) as f:
				first_timepoint_time = int(f.readline())
		self.assertEqual(first_timepoint_time, analysis_config.first_timepoint_time)
	
	def _check_single_pos_properties_file(self, analysis_config, xy_pos_idx):
		'''Check tracked colony properties file'''
		if analysis_config.xy_position_idx != xy_pos_idx:
			raise ValueError(
				'Calling analysis config without first setting xy position')
		self.assertTrue(os.path.isfile(
			analysis_config.tracked_properties_write_path))
		expected_tracked_properties_write_path = \
			os.path.join(
				self.expected_positionwise_colony_prop_dir,
				os.path.basename(analysis_config.tracked_properties_write_path)
				)
		self._compare_dataframes(
			expected_tracked_properties_write_path,
			analysis_config.tracked_properties_write_path
			)

	def _check_single_pos_output(self, xy_pos_idx):
		'''
		Check that expected output files exist (and are correct) for
		imaging field xy_pos_idx
		'''
		for phase in self.phases:
			phase_dir = self.analysis_config_obj_df.at[phase, 'phase_dir']
			expected_phase_dir = \
				self.analysis_config_obj_df.at[phase, 'expected_phase_dir']
			threshold_info = \
				self.analysis_config_obj_df.at[phase, 'threshold_info_df']
			analysis_config = \
				self.analysis_config_obj_df.at[phase, 'analysis_config']
			analysis_config.set_xy_position(xy_pos_idx)
			# check for existance of tracked colony properties file
			# NB: having this line in the loop is redundant, since this
			# file should exist in the output directory above the phase
			# directories
			self._check_single_pos_properties_file(analysis_config, xy_pos_idx)
			for t in analysis_config.timepoint_list:
				# get input im path and image name for main channel
				input_filepath, main_channel_im_name = \
					analysis_config._generate_filename(
						t, xy_pos_idx, analysis_config.main_channel_label)
				# if input image exists, check existance of ouptut images
				if os.path.isfile(input_filepath):
					# check for jpg images
					self.assertTrue(
						os.path.isfile(
							os.path.join(
								phase_dir,
								'jpgGRimages',
								(main_channel_im_name + '.jpg')
								)
							)
						)
					# check for colony mask images
					self.assertTrue(
						os.path.isfile(
							os.path.join(
								phase_dir,
								'colony_masks',
								(main_channel_im_name + '.tif')
								)
							)
						)
					# check for files in folders for 'extended display'
					if xy_pos_idx in self.extended_display_positions:
						# check existance in current threshold_info file
						self.assertTrue(main_channel_im_name in threshold_info.index)
						# check threshold file
						self.assertTrue(
							os.path.isfile(
								os.path.join(
									phase_dir,
									'threshold_plots',
									(main_channel_im_name +
										'_threshold_plot.png')
									)
								)
							)
						# check boundary im
						self.assertTrue(
							os.path.isfile(
								os.path.join(
									phase_dir,
									'boundary_ims',
									(main_channel_im_name + '.jpg')
									)
								)
							)
						# check colony center overlay
						self.assertTrue(
							os.path.isfile(
								os.path.join(
									phase_dir,
									'colony_center_overlays',
									(main_channel_im_name + '.jpg')
									)
								)
							)

	def check_combined_outputs(self):
		'''
		Check combined colony tracking outputs, growth rate analysis,
		and positionwise_colony_property_matrix dataframes against
		expected values
		'''
		# get list of files to check in parent directory
		files_to_check = ['growth_rates_combined.csv',
			'colony_properties_combined.csv']
		for phase_dir in self.analysis_config_obj_df.phase_dir:
			phase_base = os.path.basename(phase_dir)
			files_to_check.append(
				os.path.join(phase_base,'growth_rates.csv')
				)
			positionwise_colony_property_matrix_files = \
				os.listdir(os.path.join(phase_dir,
					'positionwise_colony_property_matrices'))
			matrix_files_to_add = [
				os.path.join(
					phase_base,
					'positionwise_colony_property_matrices',
					f
					)
				for f in positionwise_colony_property_matrix_files
				]
			files_to_check.extend(matrix_files_to_add)
		# check that files_to_check contain identical dataframes in
		# output_path and expected_output_path
		for f in files_to_check:
			self._compare_dataframes(
				os.path.join(self.output_path, f),
				os.path.join(self.expected_output_path, f)
				)

	def check_output(self, config_filepath, expected_output_path, single_pos = None):
		self._read_config(config_filepath, expected_output_path)
		self._check_directories()
		self._check_first_tp()
		# check correct output for every position
		if single_pos == None:
			# check for outputs of every individual position
			for xy_pos_idx in self.xy_position_vector:
				self._check_single_pos_output(xy_pos_idx)
			# check for combined tracking, growth rate, and positionwise
			# property matrix outputs
			self.check_combined_outputs()
		else:
			# check for outputs of just single_pos
			if isinstance(single_pos, int):
				self._check_single_pos_output(single_pos)
			else:
				raise TypeError('single_pos must be either None or an integer')



