#!/usr/bin/python

import unittest
import numpy as np
import pandas as pd
from PIE.image_properties import _ColonyPropertyFinder
from numpy.testing import assert_array_equal, assert_allclose
from pandas.testing import assert_frame_equal

colony_mask = np.array([
	[0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 1, 1, 1, 0],
	[0, 0, 0, 0, 1, 1, 1, 0],
	[0, 1, 1, 1, 0, 1, 1, 0],
	[0, 1, 0, 1, 0, 1, 0, 0],
	[0, 1, 0, 1, 0, 0, 0, 0],
	[0, 1, 1, 1, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0]], dtype = bool)

single_colony_mask = np.array([
	[0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0],
	[0, 1, 1, 1, 0, 0, 0, 0],
	[0, 1, 0, 1, 0, 0, 0, 0],
	[0, 1, 0, 1, 0, 0, 0, 0],
	[0, 1, 1, 1, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0]], dtype = bool)

class TestFindConnectedComponents(unittest.TestCase):

	def setUp(self):
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask)

	def test_find_connected_components(self):
		'''
		Test that colony properties identified correctly
		'''
		expected_label_num = 3
		expected_labeled_mask = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 0],
			[0, 0, 0, 0, 1, 1, 1, 0],
			[0, 2, 2, 2, 0, 1, 1, 0],
			[0, 2, 0, 2, 0, 1, 0, 0],
			[0, 2, 0, 2, 0, 0, 0, 0],
			[0, 2, 2, 2, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0]])
		# left top width height area
		expected_stat_matrix = np.array([
			[0, 0, 8, 8, 45],
			[4, 1, 3, 4, 9],
			[1, 3, 3, 4, 10]])
		# centroids are mean of pixel indices
		expected_centroids = np.array([
			[158.0/45, 160.0/45],
			[5+1.0/9, 2+1.0/9],
			[2, 4.5]])
		# run connected component finder
		self.col_prop_finder._find_connected_components()
		# check properties
		assert_array_equal(expected_labeled_mask,
			self.col_prop_finder._labeled_mask)
		self.assertEqual(expected_label_num,
			self.col_prop_finder._label_num)
		assert_array_equal(expected_stat_matrix,
			self.col_prop_finder._stat_matrix)
		assert_allclose(expected_centroids,
			self.col_prop_finder._centroids)

class TestFindAreas(unittest.TestCase):

	def test_find_areas(self):
		'''
		Tests that correct non-background areas saved into property_df
		'''
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask)
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_areas()
		expected_property_df = pd.DataFrame({'Area': [9, 10]}, dtype = 'int32')
		assert_frame_equal(expected_property_df,
			self.col_prop_finder.property_df)

	def test_find_areas_0(self):
		'''
		Tests that no areas returned for empty colony mask
		'''
		self.col_prop_finder = \
			_ColonyPropertyFinder(np.zeros(colony_mask.shape,
				dtype = colony_mask.dtype))
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_areas()
		expected_property_df = pd.DataFrame({'Area': []}, dtype = 'int32')
		assert_frame_equal(expected_property_df,
			self.col_prop_finder.property_df)

class TestFindCentroids(unittest.TestCase):

	def test_find_centroids(self):
		'''
		Tests that correct non-background centroids saved into
		property_df
		'''
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask)
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_centroids()
		expected_property_df = \
			pd.DataFrame({'cX': [5+1.0/9, 2], 'cY': [2+1.0/9, 4.5]})
		assert_frame_equal(expected_property_df,
			self.col_prop_finder.property_df)

	def test_find_centroids_0(self):
		'''
		Tests that no centroids returned for empty colony mask
		'''
		self.col_prop_finder = \
			_ColonyPropertyFinder(np.zeros(colony_mask.shape,
				dtype = colony_mask.dtype))
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_centroids()
		expected_property_df = \
			pd.DataFrame({'cX': [], 'cY': []})
		assert_frame_equal(expected_property_df,
			self.col_prop_finder.property_df)

class TestFindBoundingBox(unittest.TestCase):

	def test_find_bounding_box(self):
		'''
		Tests that correct non-background bounding_boxes saved into
		property_df
		'''
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask)
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_bounding_box()
		expected_property_df = \
			pd.DataFrame({'bb_x_left': [4, 1],
				'bb_y_top': [1, 3],
				'bb_width': [3, 3],
				'bb_height': [4, 4]}).astype('int32')
		assert_frame_equal(expected_property_df,
			self.col_prop_finder.property_df, check_like = True)

	def test_find_bounding_box_0(self):
		'''
		Tests that no bounding_boxes returned for empty colony mask
		'''
		self.col_prop_finder = \
			_ColonyPropertyFinder(np.zeros(colony_mask.shape,
				dtype = colony_mask.dtype))
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_bounding_box()
		expected_property_df = \
			pd.DataFrame({'bb_x_left': [], 'bb_y_top': [], 'bb_width': [],
				'bb_height': []}).astype('int32')
		assert_frame_equal(expected_property_df,
			self.col_prop_finder.property_df, check_like = True)

class TestFindFlatCoordinates(unittest.TestCase):

	def test_find_flat_coordinates(self):
		'''
		Tests that flat list of indices identified can be used to
		recapitulate original mask
		'''
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask)
		test_flat_coords = \
			self.col_prop_finder._find_flat_coordinates(single_colony_mask)
		test_mask = np.zeros(single_colony_mask.shape,
			dtype = single_colony_mask.dtype)
		test_mask.flat[np.fromstring(test_flat_coords,
			dtype = int, sep = ' ')] = True
		assert_array_equal(single_colony_mask, test_mask)

class TestFindPerimeter(unittest.TestCase):

	def test_find_perimeter(self):
		'''
		Tests that correct external colony perimeter returned
		'''
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask)
		expected_perimeter = 10.0
		test_perimeter = \
			self.col_prop_finder._find_perimeter(single_colony_mask)
		self.assertEqual(expected_perimeter, test_perimeter)

class TestFindColonyWiseProperties(unittest.TestCase):

	def test_find_colonywise_properties(self):
		'''
		Tests that correct perimeter and colony pixel index string
		identified
		'''
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask)
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_colonywise_properties()
		# perimeter value is counterintuitive here but makes sense when
		# you realize contour tracks left to right first
		expected_property_df = \
			pd.DataFrame({'Perimeter': [6+np.sqrt(2)+np.sqrt(2), 10.0],
				'PixelIdxList': ['12 13 14 20 21 22 29 30 37',
					'25 26 27 33 35 41 43 49 50 51']})
		assert_frame_equal(expected_property_df,
			self.col_prop_finder.property_df)

	def test_find_colonywise_properties_0(self):
		'''
		Tests that no centroids returned for empty colony mask
		'''
		pass
		self.col_prop_finder = \
			_ColonyPropertyFinder(np.zeros(colony_mask.shape,
				dtype = colony_mask.dtype))
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_centroids()
		expected_property_df = \
			pd.DataFrame({'cx': [], 'cy': []})
		#assert_frame_equal(expected_property_df,
		#	self.col_prop_finder.property_df)