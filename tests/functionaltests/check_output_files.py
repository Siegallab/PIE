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
	try:
		os.mkdir(new_output_path)
	except:
		pass
	analysis_config_file_standin = os.path.join(new_output_path, 'temp_setup.csv')
	analysis_config_df = pd.read_csv(config_file_path)
	# set output path
	analysis_config_df.loc[analysis_config_df.Parameter == 'output_path','Value'] = \
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
		self.expected_positionwise_colony_prop_dir = \
			os.path.join(self.expected_output_path,
				'positionwise_colony_properties')
		self.max_xy_position_num = \
			analysis_config_file_processor.max_xy_position_num
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
					'threshold_plots', 'threshold_info_comb.csv'
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
	
	def _read_in_df(self, df_path, index_col = None):
		'''
		Returns dataframe at df_path, which can be either parquet or csv
		'''
		_, ext = os.path.splitext(df_path)
		if ext == '.csv':
			df = pd.read_csv(df_path, dtype={'area': 'int32'}, index_col = index_col)
		elif ext == '.parquet':
			df = pd.read_parquet(df_path)
		else:
			raise ValueError(
				'Expecting parquet or csv file, got extension ' + ext)
		return(df)

	def _compare_dataframes(
		self, expected_df_path, test_df_path, index_col = None
		):
		'''
		Reads and compares dataframe in expected_df_path to test_df_path
		'''
		expected_df = self._read_in_df(expected_df_path, index_col = index_col)
		test_df = self._read_in_df(test_df_path, index_col = index_col)
		assert_frame_equal(expected_df, test_df, check_like = True)

	def _check_directories(self):
		'''
		Check that subdirectories for each phase exist
		'''
		self.assertTrue(os.path.isdir(self.output_path))
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

	def _check_single_pos_output(self, xy_pos_idx, single_pos_only = False):
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
			if single_pos_only:
				self._check_single_pos_properties_file(
					analysis_config, xy_pos_idx
					)
			for t in analysis_config.timepoint_list:
				# get input im path and image name for main channel
				input_filepath, main_channel_im_name = \
					analysis_config.generate_filename(
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
						if single_pos_only:
							# check existance in current threshold_info file
							self.assertTrue(
								main_channel_im_name in threshold_info.index or
								os.path.isfile(
									os.path.join(
										phase_dir,
										'threshold_plots',
										(
											'threshold_info_' + 
											main_channel_im_name + 
											'.csv'
											)
										)
									)
								)
						# check threshold files
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

	def _check_comb_gr_df(self):
		'''
		Check equivalence of combined growth rate data frames
		'''
		self._compare_dataframes(
				os.path.join(self.output_path, 'growth_rates_combined.csv'),
				os.path.join(
					self.expected_output_path, 'growth_rates_combined.csv'
					)
				)

	def _check_col_props(self):
		'''
		Check equivalence of combined colony property data frames
		'''
		self._compare_dataframes(
				os.path.join(self.output_path, 'colony_properties_combined.csv'),
				os.path.join(
					self.expected_output_path, 'colony_properties_combined.csv'
					)
				)

	def _check_phase_gr_df(self, phase_base_dir):
		'''
		Check equivalence of single-phase growth rate data frames
		'''
		self._compare_dataframes(
				os.path.join(self.output_path, phase_base_dir, 'growth_rates.csv'),
				os.path.join(
					self.expected_output_path, phase_base_dir, 'growth_rates.csv'
					)
				)

	def _check_phase_filt_col_df(self, phase_base_dir):
		'''
		Check equivalence of single-phase filtered colony data frames
		'''
		self._compare_dataframes(
				os.path.join(self.output_path, phase_base_dir, 'filtered_colonies.csv'),
				os.path.join(
					self.expected_output_path, phase_base_dir, 'filtered_colonies.csv'
					)
				)

	def _check_phase_thresh_df(self, phase_base_dir):
		'''
		Check equivalence of single-phase combined threshold info data 
		frames
		'''
		self._compare_dataframes(
				os.path.join(
					self.output_path,
					phase_base_dir,
					'threshold_plots',
					'threshold_info_comb.csv'
					),
				os.path.join(
					self.expected_output_path,
					phase_base_dir,
					'threshold_plots',
					'threshold_info_comb.csv'
					),
				index_col = 0
				)
	def _check_positionwise_col_prop_mat_list(
		self, expected_prop_list, phase_base_dir
		):
		'''
		Check that files exist for full list of expected colony properties
		'''
		expected_positionwise_colony_property_matrix_files = \
			{ x + '_property_mat.csv' for x in expected_prop_list }
		test_positionwise_colony_property_matrix_files = \
				set(os.listdir(os.path.join(
					self.output_path,
					phase_base_dir,
					'positionwise_colony_property_matrices'
					)))
		self.assertEqual(
			expected_positionwise_colony_property_matrix_files,
			test_positionwise_colony_property_matrix_files
			)

	def _check_positionwise_col_prop_mat(
		self, prop, phase_base_dir
		):
		'''
		Check single-phase positionwise colony property matrix for 
		property 'prop'
		'''
		expected_positionwise_colony_property_matrix_file = \
			os.path.join(
				self.expected_output_path,
				phase_base_dir,
				'positionwise_colony_property_matrices',
				prop + '_property_mat.csv'
				)
		test_positionwise_colony_property_matrix_file = \
			os.path.join(
				self.output_path,
				phase_base_dir,
				'positionwise_colony_property_matrices',
				prop + '_property_mat.csv'
				)
		self._compare_dataframes(
			expected_positionwise_colony_property_matrix_file,
			test_positionwise_colony_property_matrix_file
			)

	def check_combined_outputs(self):
		'''
		Check combined colony tracking outputs, growth rate analysis,
		and positionwise_colony_property_matrix dataframes against
		expected values
		'''
		# get list of files to check in parent directory
		### CURRENTLY NOT CHECKING FOR EXTRA FILES
		self._check_comb_gr_df()
		self._check_col_props()

		for phase_dir in self.analysis_config_obj_df.phase_dir:
			phase_base_dir = os.path.basename(phase_dir)
			self._check_phase_gr_df(phase_base_dir)
			self._check_phase_filt_col_df(phase_base_dir)
			self._check_phase_thresh_df(phase_base_dir)
			# check for all positionwise colony matrix files based on 
			# columns of colony_properties_combined.csv
			# (no standard for what these will be due to diff possible 
			# fluor channels, etc)
			###
			# colony properties for which to NOT make property matrices
			# (should be same as list in 
			# PIE.colony_prop_compilation.CompileColonyData)
			cols_to_exclude = ['timepoint', 'phase_num', 'xy_pos_idx',
				'time_tracking_id', 'main_image_name', 'bb_height', 'bb_width',
				'bb_x_left', 'bb_y_top', 'cross_phase_tracking_id']
			colony_data_tracked_df = \
				self._read_in_df(
					os.path.join(self.output_path, 'colony_properties_combined.csv')
					)
			expected_prop_list = \
				list(set(colony_data_tracked_df.columns.to_list()) -
					set(cols_to_exclude))
			self._check_positionwise_col_prop_mat_list(
				expected_prop_list, phase_base_dir
				)
			# loop through expected properties and compare property dfs
			for curr_prop in expected_prop_list:
				self._check_positionwise_col_prop_mat(curr_prop, phase_base_dir)

	def _check_movies(self):
		'''
		Checks movie outputs for extended_display_positions
		(which only created after full processing)
		'''
		for xy_pos_idx in self.extended_display_positions:
			# check movie
			self.assertTrue(
				os.path.isfile(
					os.path.join(
						self.output_path,
						'movies',
						('xy'+str(xy_pos_idx)+'_growing_colonies_movie.gif')
						)
					)
				)

	def check_output(self, config_filepath, expected_output_path, single_pos = None):
		self._read_config(config_filepath, expected_output_path)
		self._check_directories()
		self._check_first_tp()
		# check correct output for every position
		if single_pos is None:
			self._check_movies()
			# check for outputs of every individual position
			for xy_pos_idx in self.xy_position_vector:
				self._check_single_pos_output(xy_pos_idx)
			# check for combined tracking, growth rate, and positionwise
			# property matrix outputs
			self.check_combined_outputs()
		else:
			# check for outputs of just single_pos
			if isinstance(single_pos, int):
				self._check_single_pos_output(single_pos, single_pos_only = True)
			else:
				raise TypeError('single_pos must be either None or an integer')



