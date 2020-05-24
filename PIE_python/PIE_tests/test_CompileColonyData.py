#!/usr/bin/python

import os
import numpy as np
import pandas as pd
import unittest
from PIE.growth_measurement import _CompileColonyData
from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal

phase_tracked_properties_df = pd.DataFrame({
	'unique_tracking_id':
		['growth_1_col2', 'growth_1_col1', 'growth_1_col1', 'growth_1_col2',
			'growth_1_col1'],
	'timepoint': [1, 1, 3, 4, 4],
	'area': [100, 92, 150, 205, 140],
	})

class Test_GetIndexLocations(unittest.TestCase):

	def setUp(self):
		self.colony_data_compiler = object.__new__(_CompileColonyData)
		self.colony_data_compiler.colony_data_tracked_df = \
			phase_tracked_properties_df

	def test_get_index_locations(self):
		expected_timepoint_list = np.array([1,3,4])
		expected_timepoint_indices = np.array([0,0,1,2,2])
		expected_unique_tracking_id_list = \
			np.array(['growth_1_col1', 'growth_1_col2'])
		expected_unique_tracking_id_indices = \
			np.array([1, 0, 0, 1, 0])
		expected_empty_col_property_mat = np.array([
			[np.nan, np.nan, np.nan],
			[np.nan, np.nan, np.nan]])
		self.colony_data_compiler._get_index_locations()
		assert_equal(expected_timepoint_list,
			self.colony_data_compiler.timepoint_list)
		assert_equal(expected_timepoint_indices,
			self.colony_data_compiler.timepoint_indices)
		assert_equal(expected_unique_tracking_id_list,
			self.colony_data_compiler.unique_tracking_id_list)
		assert_equal(expected_unique_tracking_id_indices,
			self.colony_data_compiler.unique_tracking_id_indices)
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
				[92.0, 150.0, 140.0],
				[100.0, np.nan, 205.0]]),
			index = ['growth_1_col1', 'growth_1_col2'],
			columns = [1,3,4])
		test_col_property_df = \
			self.colony_data_compiler._create_property_mat('area')
		assert_frame_equal(expected_col_property_df, test_col_property_df)

