#!/usr/bin/python

import os
import shutil
import unittest
import PIE
from test.functionaltests.check_output_files import OutputChecker, make_setup_filepath_standin
from io import StringIO

output_path = os.path.join('test','test_functional_output_dir')

class ParentTestClass(object):
	'''
	Parent test for running identical tests using either python or
	commandline
	'''

	def test_existing_pos_singlephase(self):
		'''
		Tests running image analysis and single colony tracking on existing
		position in single-phase experiment
		'''
		expected_output_path = os.path.join(
			'test',
			'expected_ft_outputs',
			'SL_170619_2_GR_small_expected_out'
			)
		xy_pos = 11;
		analysis_config_filepath = \
			os.path.join('sample_PIE_setup_files','gr_phase_setup.csv')
		self._run_test_single_pos(
			analysis_config_filepath,
			xy_pos,
			expected_output_path)

	def test_nonexisting_pos_singlephase(self):
		'''
		Tests running image analysis and single colony tracking on existing
		position in single-phase experiment
		'''
		expected_output_path = os.path.join(
			'test',
			'expected_ft_outputs',
			'SL_170619_2_GR_small_expected_out'
			)
		xy_pos = 15
		analysis_config_filepath = \
			os.path.join('sample_PIE_setup_files','gr_phase_setup.csv')
		self._run_test_single_pos(
			analysis_config_filepath,
			xy_pos,
			expected_output_path)

	def test_existing_pos_twophase(self):
		'''
		Tests running image analysis and single colony tracking on existing
		position in two-phase experiment
		'''
		expected_output_path = os.path.join(
			'test',
			'expected_ft_outputs',
			'SL_180519_small_expected_out'
			)
		xy_pos = 401
		analysis_config_filepath = \
			os.path.join('sample_PIE_setup_files','two_phase_setup.csv')
		self._run_test_single_pos(
			analysis_config_filepath,
			xy_pos,
			expected_output_path)

	def test_existing_pos_postphase(self):
		'''
		Tests running image analysis and single colony tracking on existing
		position in two-phase experiment
		'''
		expected_output_path = os.path.join(
			'test',
			'expected_ft_outputs',
			'EP_170202_small_expected_out'
			)
		xy_pos = 735
		analysis_config_filepath = \
			os.path.join('sample_PIE_setup_files','gr_with_postfluor_setup.csv')
		self._run_test_single_pos(
			analysis_config_filepath,
			xy_pos,
			expected_output_path)

class SetupTeardown(unittest.TestCase):
	def setUp(self):
		self.output_checker = OutputChecker()

	def tearDown(self):
		# remove directories created
		shutil.rmtree(output_path)

class TestPythonRun(SetupTeardown, ParentTestClass):
	'''
	Test running image analysis + colony tracking on single position in
	python
	'''

	def _run_test_single_pos(self, analysis_config_filepath, xy_pos,
		expected_output_path):
		'''
		Test running full experiment, both using analysis_config_file
		and the object dataframe derived from it
		'''
		self._run_config_file_test(
			analysis_config_filepath,
			xy_pos,
			expected_output_path
			)
		shutil.rmtree(output_path)
		self._run_analysis_obj_df_test(
			analysis_config_filepath,
			xy_pos,
			expected_output_path
			)

	def _run_config_file_test(self, analysis_config_filepath, xy_pos,
		expected_output_path):
		'''
		Uses analysis_config_file to test tracking of xy_pos
		'''
		analysis_config_file_standin = \
			make_setup_filepath_standin(
				analysis_config_filepath,
				output_path)
		PIE.track_colonies_single_pos(
			xy_pos,
			analysis_config_file = analysis_config_file_standin
			)
		self.output_checker.check_output(
			analysis_config_file_standin,
			expected_output_path,
			single_pos = xy_pos
			)

	def _run_analysis_obj_df_test(self, analysis_config_filepath, xy_pos,
		expected_output_path):
		'''
		Uses analysis_config_file to test tracking of xy_pos
		'''
		analysis_config_file_standin = \
			make_setup_filepath_standin(
				analysis_config_filepath,
				output_path)
		config_df = PIE.process_setup_file(analysis_config_file_standin)
		PIE.track_colonies_single_pos(
			xy_pos,
			analysis_config_obj_df = config_df
			)
		self.output_checker.check_output(
			analysis_config_file_standin,
			expected_output_path,
			single_pos = xy_pos
			)

class TestCommandlineRun(SetupTeardown, ParentTestClass):
	'''
	Test running image analysis + colony tracking on single position in
	commandline
	'''

	def _run_test_single_pos(self, analysis_config_filepath, xy_pos,
		expected_output_path):
		'''
		Uses analysis_config_file to test tracking of xy_pos
		'''
		analysis_config_file_standin = \
			make_setup_filepath_standin(
				analysis_config_filepath,
				output_path)
		submission_string = ' '.join([
			'pie track_single_pos',
			str(xy_pos),
			str(analysis_config_file_standin)])
		os.system(submission_string)
		self.output_checker.check_output(
			analysis_config_file_standin,
			expected_output_path,
			single_pos = xy_pos
			)
