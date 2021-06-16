#!/usr/bin/python

import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal
from PIE import colony_filters, analysis_configuration

areas_df = pd.DataFrame(
	np.array(
		[[1, 2, 1.5, 4],
		[2, 5, 15, 16],
		[np.nan, np.nan, 1, 2],
		[0, 3, 8, 7.9],
		[1, 1, 1, 2]]),
	index = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5'],
	columns = [1,2,3,4])

class TestFilterByMaxAreaPixelDecrease(unittest.TestCase):
	'''
	Tests filtering by max area pixel decrease
	'''

	def setUp(self):
		self.analysis_config = object.__new__(analysis_configuration.AnalysisConfig)

	def test_filter_by_max_area_pixel_decrease_small(self):
		'''
		Test allowing areas to decrease by a small amount
		'''
		self.analysis_config.max_area_pixel_decrease = 0.4
		filter_obj = colony_filters._FilterByMaxAreaPixelDecrease(areas_df,
			self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.array([
			[1, 1, 0, 0],
			[1, 1, 1, 1],
			[1, 1, 1, 1],
			[1, 1, 1, 1],
			[1, 1, 1, 1]], dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

	def test_filter_by_max_area_pixel_decrease_zero(self):
		'''
		Test not allowing areas to decrease by any amount
		'''
		self.analysis_config.max_area_pixel_decrease = 0
		filter_obj = colony_filters._FilterByMaxAreaPixelDecrease(areas_df,
			self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.array([
			[1, 1, 0, 0],
			[1, 1, 1, 1],
			[1, 1, 1, 1],
			[1, 1, 1, 0],
			[1, 1, 1, 1]], dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

	def test_filter_by_max_area_pixel_decrease_inf(self):
		'''
		Test allowing areas to decrease by any amount
		'''
		self.analysis_config.max_area_pixel_decrease = np.inf
		filter_obj = colony_filters._FilterByMaxAreaPixelDecrease(areas_df,
			self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.ones(areas_df.shape, dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

class test_filter_data(unittest.TestCase):
	'''
	Tests generic data filtration
	'''

	def _test_filter_max_area_fold_increase(self):
		'''
		Tests filtration by max_area_fold_increase
		'''
		filtration_type = 'max_area_fold_increase'
		max_area_fold_increase = 2.5
		expected_filter_pass_bool = np.array([
			[1, 1, 1, 0],
			[1, 1, 0, 0],
			[1, 1, 1, 1],
			[1, 0, 0, 0],
			[1, 1, 1, 1]], dtype = bool)
		expected_removed_locations = [('col_1', 4), ('col_2', 3), ('col_4', 2)]
		test_filter_pass_bool, test_removed_locations = \
			colony_filters.filter_data(filtration_type, areas_df,
				max_area_fold_increase)
		self.assertEqual(expected_removed_locations, test_removed_locations)
		assert_equal(expected_filter_pass_bool, test_filter_pass_bool)



class test_filter_by_max_area_fold_decrease(unittest.TestCase):
	'''
	Tests filtering by max area fold decrease
	'''

	def setUp(self):
		self.analysis_config = object.__new__(analysis_configuration.AnalysisConfig)

	def test_filter_by_max_area_fold_decrease_small(self):
		'''
		Test allowing areas to decrease by a small amount
		'''
		self.analysis_config.max_area_fold_decrease = 1.2
		filter_obj = colony_filters._FilterByMaxAreaFoldDecrease(areas_df,
			self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.array([
			[1, 1, 0, 0],
			[1, 1, 1, 1],
			[1, 1, 1, 1],
			[1, 1, 1, 1],
			[1, 1, 1, 1]], dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

	def test_filter_by_max_area_fold_decrease_very_small(self):
		'''
		Test not allowing areas to decrease by another small amount
		'''
		self.analysis_config.max_area_fold_decrease = 1.01
		filter_obj = colony_filters._FilterByMaxAreaFoldDecrease(areas_df,
			self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.array([
			[1, 1, 0, 0],
			[1, 1, 1, 1],
			[1, 1, 1, 1],
			[1, 1, 1, 0],
			[1, 1, 1, 1]], dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

	def test_filter_by_max_area_fold_decrease_one(self):
		'''
		Test not allowing areas to decrease by any amount
		'''
		self.analysis_config.max_area_fold_decrease = 1
		filter_obj = colony_filters._FilterByMaxAreaFoldDecrease(areas_df,
			self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.array([
			[1, 1, 0, 0],
			[1, 1, 1, 1],
			[1, 1, 1, 1],
			[1, 1, 1, 0],
			[1, 1, 1, 1]], dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

	def test_filter_by_max_area_fold_decrease_inf(self):
		'''
		Test allowing areas to decrease by any amount
		'''
		self.analysis_config.max_area_fold_decrease = np.inf
		filter_obj = colony_filters._FilterByMaxAreaFoldDecrease(areas_df,
			self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.ones(areas_df.shape, dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)
		
class test_filter_by_max_area_fold_increase(unittest.TestCase):
	'''
	Tests filtering by max area fold increase
	'''

	def setUp(self):
		self.analysis_config = object.__new__(analysis_configuration.AnalysisConfig)

	def test_filter_by_max_area_fol_increase_small(self):
		'''
		Test allowing areas to increase by a small amount
		'''
		self.analysis_config.max_area_fold_increase = 2.5
		filter_obj = colony_filters._FilterByMaxAreaFoldIncrease(areas_df,
			self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.array([
			[1, 1, 1, 0],
			[1, 1, 0, 0],
			[1, 1, 1, 1],
			[1, 0, 0, 0],
			[1, 1, 1, 1]], dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

	def test_filter_by_max_area_fold_increase_one(self):
		'''
		Test not allowing areas to increase by any amount
		'''
		self.analysis_config.max_area_fold_increase = 1
		filter_obj = colony_filters._FilterByMaxAreaFoldIncrease(areas_df,
			self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.array([
			[1, 0, 0, 0],
			[1, 0, 0, 0],
			[1, 1, 1, 0],
			[1, 0, 0, 0],
			[1, 1, 1, 0]], dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

	def test_filter_by_max_area_fold_increase_inf(self):
		'''
		Test allowing areas to increase by any amount
		'''
		self.analysis_config.max_area_fold_increase = np.inf
		filter_obj = colony_filters._FilterByMaxAreaFoldIncrease(areas_df,
			self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.ones(areas_df.shape, dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

class TestFilterByGrowthWindowTimepoints(unittest.TestCase):
	'''
	Tests filtering by growth window timepoints
	'''

	def setUp(self):
		self.analysis_config = object.__new__(analysis_configuration.AnalysisConfig)
		self.areas_with_nulls_df = pd.DataFrame(np.array([
			[4, 6, 9, np.nan, np.nan, 11, 14, np.nan],
			[np.nan, np.nan, 2, 3, 5, 8, 11, np.nan],
			[1, 1, 2, 3, 5, 8, 12, 14]]),
		index = ['col_1', 'col_2', 'col_4'],
		columns = [1,2,3,4,5,6,7,8])

	def test_window_size_2(self):
		'''
		Tests growth_window_timepoints of 2, which results in everything
		passing the filter
		'''
		self.analysis_config.growth_window_timepoints = 2
		filter_obj = colony_filters._FilterByGrowthWindowTimepoints(
			self.areas_with_nulls_df, self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.array([
			[1, 1, 1, 0, 0, 1, 1, 0],
			[0, 0, 1, 1, 1, 1, 1, 0],
			[1, 1, 1, 1, 1, 1, 1, 1]], dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

	def test_window_size_3(self):
		'''
		Tests growth_window_timepoints of 3, which removes only two
		timepoints from the row corresponding to col_1
		'''
		self.analysis_config.growth_window_timepoints = 3
		filter_obj = colony_filters._FilterByGrowthWindowTimepoints(
			self.areas_with_nulls_df, self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.array([
			[1, 1, 1, 0, 0, 0, 0, 0],
			[0, 0, 1, 1, 1, 1, 1, 0],
			[1, 1, 1, 1, 1, 1, 1, 1]], dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

	def test_window_size_8(self):
		'''
		Tests growth_window_timepoints of 8, which removes everything
		except the last row
		'''
		self.analysis_config.growth_window_timepoints = 8
		filter_obj = colony_filters._FilterByGrowthWindowTimepoints(
			self.areas_with_nulls_df, self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[1, 1, 1, 1, 1, 1, 1, 1]], dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

	def test_window_size_8(self):
		'''
		Tests growth_window_timepoints of 20, which removes everything
		'''
		self.analysis_config.growth_window_timepoints = 20
		filter_obj = colony_filters._FilterByGrowthWindowTimepoints(
			self.areas_with_nulls_df, self.analysis_config)
		test_filter_bool = filter_obj._filtration_method()
		expected_filter_bool = \
			np.zeros(self.areas_with_nulls_df.shape, dtype = bool)
		assert_equal(expected_filter_bool, test_filter_bool)

class TestIdFilteredLocations(unittest.TestCase):
	'''
	Tests that correct tuples of areas_df index and column name is
	returned for a given filter_pass bool matrix
	'''

	def setUp(self):
		self.analysis_config = object.__new__(analysis_configuration.AnalysisConfig)
		self.filter_obj = \
			colony_filters._FilterBaseClass(areas_df, self.analysis_config)

	def test_id_filtered_locations_simple(self):
		filter_bool = np.array([
			[1, 0, 0, 0],
			[1, 1, 0, 1],
			[1, 1, 1, 0],
			[1, 1, 1, 1],
			[0, 0, 0, 1]], dtype = bool)
		expected_filtered_locations = \
			pd.DataFrame(np.array([2,3,4,1]),
				index = ['col_1', 'col_2', 'col_3', 'col_5'],
				columns = ['filtered_columns'])
		test_filtered_locations = \
			self.filter_obj._id_removed_locations(filter_bool)
		assert_frame_equal(expected_filtered_locations,
			test_filtered_locations)

if __name__ == '__main__':
	unittest.main()