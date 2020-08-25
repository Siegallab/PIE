#!/usr/bin/python

import os
import shutil
import unittest
import PIE
import numpy as np
from PIE_tests.functionaltests.check_output_files import OutputChecker, make_setup_filepath_standin
from io import StringIO

output_path = 'PIE_tests/test_functional_output_dir'

class ParentTestClass(object):
	'''
	Parent test for running identical tests using either python or
	commandline
	'''

	def test_singlephase(self):
		'''
		Tests running full single-phase experiment
		'''
		expected_output_path = os.path.join(
			'PIE_tests',
			'expected_ft_outputs',
			'SL_170619_2_GR_small_expected_out'
			)
		analysis_config_filepath = 'sample_PIE_setup_files/gr_phase_setup.csv'
		self._run_full_analysis_test(
			analysis_config_filepath,
			expected_output_path)

	def test_singlephase_nofile(self):
		'''
		Tests running full single-phase experiment without setup file
		'''
		expected_output_path = os.path.join(
			'PIE_tests',
			'expected_ft_outputs',
			'SL_170619_2_GR_small_expected_out'
			)
		analysis_config_filepath = 'sample_PIE_setup_files/gr_phase_setup.csv'
		self._run_full_analysis_no_setup_test(expected_output_path)

	def test_twophase(self):
		'''
		Tests running full two-phase experiment
		'''
		expected_output_path = os.path.join(
			'PIE_tests',
			'expected_ft_outputs',
			'SL_180519_small_expected_out'
			)
		analysis_config_filepath = 'sample_PIE_setup_files/two_phase_setup.csv'
		self._run_full_analysis_test(
			analysis_config_filepath,
			expected_output_path)

class SetupTeardown(unittest.TestCase):
	def setUp(self):
		self.output_checker = OutputChecker()

	def tearDown(self):
		# remove directories created
		shutil.rmtree(output_path)

class TestPythonRun(SetupTeardown, ParentTestClass):
	'''
	Test running full experiment in python
	'''

	def _run_full_analysis_test(self, analysis_config_filepath,
		expected_output_path):
		'''
		Test running full experiment, both using analysis_config_file
		and the object dataframe derived from it
		'''
		self._run_config_file_test(
			analysis_config_filepath,
			expected_output_path
			)
		shutil.rmtree(output_path)
		self.output_checker = OutputChecker()
		self._run_analysis_obj_df_test(
			analysis_config_filepath,
			expected_output_path
			)

	def _run_config_file_test(self, analysis_config_filepath,
		expected_output_path):
		'''
		Uses analysis_config_file to run full experiment
		'''
		analysis_config_file_standin = \
			make_setup_filepath_standin(
				analysis_config_filepath,
				output_path)
		PIE.run_growth_rate_analysis(
			analysis_config_file = analysis_config_file_standin
			)
		self.output_checker.check_output(
			analysis_config_file_standin,
			expected_output_path
			)

	def _run_analysis_obj_df_test(self, analysis_config_filepath,
		expected_output_path):
		'''
		Uses analysis_object_df to run full experiment
		'''
		analysis_config_file_standin = \
			make_setup_filepath_standin(
				analysis_config_filepath,
				output_path)
		config_df = PIE.process_setup_file(analysis_config_file_standin)
		PIE.run_growth_rate_analysis(
			analysis_config_obj_df = config_df
			)
		self.output_checker.check_output(
			analysis_config_file_standin,
			expected_output_path
			)

	def _run_full_analysis_no_setup_test(self, expected_output_path):
		'''
		Runs default analysis using gr_phase_setup setup file with
		modified options
		'''
		analysis_config_file_standin = \
			make_setup_filepath_standin(
				'sample_PIE_setup_files/gr_phase_setup.csv',
				output_path)
		input_path = os.path.join('PIE_test_data','IN','SL_170619_2_GR_small')
		PIE.run_default_growth_rate_analysis(input_path, output_path,
			total_timepoint_num = 10, total_xy_position_num = 1000,
			timepoint_spacing = 3600, extended_display_positions = [1, 4, 11],
			growth_window_timepoints = 7,
			max_area_pixel_decrease = 500, min_colony_area = 30,
			max_colony_area = np.inf, min_correlation = 0.9, min_neighbor_dist = 100)
		self.output_checker.check_output(
			analysis_config_file_standin,
			expected_output_path
			)

class TestCommandlineRun(SetupTeardown, ParentTestClass):
	'''
	Test running full experiment in commandline
	'''

	def _run_full_analysis_test(self, analysis_config_filepath,
		expected_output_path):
		'''
		Uses analysis_config_file to test tracking of xy_pos
		'''
		analysis_config_file_standin = \
			make_setup_filepath_standin(
				analysis_config_filepath,
				output_path)
		submission_string = ' '.join([
			'pie run',
			str(analysis_config_file_standin)])
		os.system(submission_string)
		self.output_checker.check_output(
			analysis_config_file_standin,
			expected_output_path
			)

	def _run_full_analysis_no_setup_test(self, expected_output_path):
		os.mkdir(output_path)
		pass
