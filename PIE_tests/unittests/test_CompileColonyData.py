#!/usr/bin/python

import os
import numpy as np
import pandas as pd
import unittest
from PIE.growth_measurement import _CompileColonyData
from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal

phase_tracked_properties_df = pd.DataFrame({
	'time_tracking_id':
		['phase_1_xy1_col2', 'phase_1_xy1_col1', 'phase_1_xy1_col1', 'phase_1_xy1_col2',
			'phase_1_xy1_col1', 'phase_1_xy3_col1', 'phase_1_xy3_col1'],
	'cross_phase_tracking_id':
		['phase_1_xy1_col2', 'phase_1_xy1_col1', 'phase_1_xy1_col1', 'phase_1_xy1_col2',
			'phase_1_xy1_col1', 'phase_1_xy3_col1', 'phase_1_xy3_col1'],
	'timepoint': [1, 1, 3, 4, 4, 3, 5],
	'area': [100, 92, 150, 205, 140, 160, 180],
	'xy_pos_idx': [1, 1, 1, 1, 1, 3, 3],
	'phase_num': \
		[1]*7
	})

class Test_GetIndexLocations(unittest.TestCase):

	def setUp(self):
		self.colony_data_compiler = object.__new__(_CompileColonyData)
		self.colony_data_compiler.colony_data_tracked_df = \
			phase_tracked_properties_df

	def test_get_index_locations(self):
		expected_timepoint_list = np.array([1,3,4,5])
		expected_timepoint_indices = np.array([0,0,1,2,2,1,3])
		expected_time_tracking_id_list = \
			np.array(['phase_1_xy1_col1', 'phase_1_xy1_col2', 'phase_1_xy3_col1'])
		expected_time_tracking_id_indices = \
			np.array([1, 0, 0, 1, 0, 2, 2])
		expected_empty_col_property_mat = np.array([
			[np.nan, np.nan, np.nan, np.nan],
			[np.nan, np.nan, np.nan, np.nan],
			[np.nan, np.nan, np.nan, np.nan]])
		self.colony_data_compiler._get_index_locations()
		assert_equal(expected_timepoint_list,
			self.colony_data_compiler.timepoint_list)
		assert_equal(expected_timepoint_indices,
			self.colony_data_compiler.timepoint_indices)
		assert_equal(expected_time_tracking_id_list,
			self.colony_data_compiler.time_tracking_id_list)
		assert_equal(expected_time_tracking_id_indices,
			self.colony_data_compiler.time_tracking_id_indices)
		assert_equal(expected_empty_col_property_mat,
			self.colony_data_compiler.empty_col_property_mat)

class Test_CreatePropertyMat(unittest.TestCase):

	def setUp(self):
		self.colony_data_compiler = object.__new__(_CompileColonyData)
		self.colony_data_compiler.colony_data_tracked_df = \
			phase_tracked_properties_df
		self.colony_data_compiler._get_index_locations()

	def test_create_property_mat_area(self):
		'''
		Tests creation of an area colony property matrix
		'''
		expected_col_property_df = pd.DataFrame(np.array([
				[92.0, 150.0, 140.0, np.nan],
				[100.0, np.nan, 205.0, np.nan],
				[np.nan, 160.0, np.nan, 180.0]]),
			index = ['phase_1_xy1_col1', 'phase_1_xy1_col2', 'phase_1_xy3_col1'],
			columns = [1,3,4,5])
		test_col_property_df = \
			self.colony_data_compiler._create_property_mat('area')
		assert_frame_equal(expected_col_property_df, test_col_property_df)

class Test_GenerateImagingInfoDf(unittest.TestCase):

	def setUp(self):
		self.colony_data_compiler = object.__new__(_CompileColonyData)
		self.colony_data_compiler.colony_data_tracked_df = \
			phase_tracked_properties_df

	def test_generate_imaging_info_df(self):
		'''
		Tests generation of imaging info df
		'''
		expected_imaging_info_df = pd.DataFrame({
				'cross_phase_tracking_id': \
					['phase_1_xy1_col2', 'phase_1_xy1_col1', 'phase_1_xy3_col1'],
				'xy_pos_idx': [1, 1, 3],
				'phase_num': [1]*3},
			index = ['phase_1_xy1_col2', 'phase_1_xy1_col1', 'phase_1_xy3_col1'])
		expected_imaging_info_df.index.name = 'time_tracking_id'
		test_imaging_info_df = \
			self.colony_data_compiler.generate_imaging_info_df()
		assert_frame_equal(expected_imaging_info_df, test_imaging_info_df)

