#!/usr/bin/python

import cv2
import numpy as np
import os
import pandas as pd
import unittest
from PIE.image_properties import _ColonyPropertyFinder
from numpy.testing import assert_array_equal, assert_allclose
from pandas.testing import assert_frame_equal

colony_mask = np.array([
	[0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 1, 1, 1, 0],
	[0, 0, 0, 0, 1, 1, 1, 0],
	[0, 1, 1, 0, 0, 1, 1, 0],
	[0, 1, 1, 1, 0, 1, 0, 0],
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

max_col_num = 5

class TestFindConnectedComponents(unittest.TestCase):

	def test_find_connected_components(self):
		'''
		Test that colony properties identified correctly
		'''
		col_prop_finder = _ColonyPropertyFinder(colony_mask, max_col_num)
		expected_label_num = 3
		expected_labeled_mask = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 0],
			[0, 0, 0, 0, 1, 1, 1, 0],
			[0, 2, 2, 0, 0, 1, 1, 0],
			[0, 2, 2, 2, 0, 1, 0, 0],
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
			[159.0/45, 159.0/45],
			[5+1.0/9, 2+1.0/9],
			[1.9, 4.6]])
		# run connected component finder
		col_prop_finder._find_connected_components()
		# check properties
		assert_array_equal(expected_labeled_mask,
			col_prop_finder.labeled_mask)
		self.assertEqual(expected_label_num,
			col_prop_finder._label_num)
		assert_array_equal(expected_stat_matrix,
			col_prop_finder._stat_matrix)
		assert_allclose(expected_centroids,
			col_prop_finder._centroids)

	def test_find_connected_components_too_many(self):
		'''
		Test that no colony properties identified when number of 
		colonies greather than max_col_num
		'''
		temp_max_col_num = 1
		col_prop_finder = _ColonyPropertyFinder(colony_mask, temp_max_col_num)
		expected_label_num = 1
		expected_labeled_mask = np.zeros_like(colony_mask)
		# left top width height area
		expected_stat_matrix = np.array([[0, 0, 8, 8, 64]])
		# centroids are mean of pixel indices
		expected_centroids = np.array([[3.5, 3.5]])
		# run connected component finder
		col_prop_finder._find_connected_components()
		# check properties
		assert_array_equal(expected_labeled_mask,
			col_prop_finder.labeled_mask)
		self.assertEqual(expected_label_num,
			col_prop_finder._label_num)
		assert_array_equal(expected_stat_matrix,
			col_prop_finder._stat_matrix)
		assert_allclose(expected_centroids,
			col_prop_finder._centroids)


class TestFindAreas(unittest.TestCase):

	def test_find_areas(self):
		'''
		Tests that correct non-background areas saved into property_df
		'''
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask, max_col_num)
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_areas()
		expected_property_df = pd.DataFrame({'area': [9, 10]}, dtype = 'int32')
		assert_frame_equal(expected_property_df,
			self.col_prop_finder.property_df)

	def test_find_areas_0(self):
		'''
		Tests that no areas returned for empty colony mask
		'''
		self.col_prop_finder = \
			_ColonyPropertyFinder(np.zeros(colony_mask.shape,
				dtype = colony_mask.dtype), max_col_num)
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_areas()
		expected_property_df = pd.DataFrame({'area': []}, dtype = 'int32')
		assert expected_property_df.equals(self.col_prop_finder.property_df)

	def test_find_areas_too_many(self):
		'''
		Tests that no areas returned if number of objects greater than 
		max_col_num
		'''
		temp_max_col_num = 1
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask, temp_max_col_num)
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_areas()
		expected_property_df = pd.DataFrame({'area': []}, dtype = 'int32')
		assert expected_property_df.equals(self.col_prop_finder.property_df)

class TestFindCentroids(unittest.TestCase):

	def test_find_centroids(self):
		'''
		Tests that correct non-background centroids saved into
		property_df
		'''
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask, max_col_num)
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_centroids()
		expected_property_df = \
			pd.DataFrame({'cX': [5+1.0/9, 1.9], 'cY': [2+1.0/9, 4.6]})
		assert_frame_equal(expected_property_df,
			self.col_prop_finder.property_df)

	def test_find_centroids_0(self):
		'''
		Tests that no centroids returned for empty colony mask
		'''
		self.col_prop_finder = \
			_ColonyPropertyFinder(np.zeros(colony_mask.shape,
				dtype = colony_mask.dtype), max_col_num)
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_centroids()
		expected_property_df = \
			pd.DataFrame({'cX': [], 'cY': []})
		assert expected_property_df.equals(self.col_prop_finder.property_df)

class TestFindBoundingBox(unittest.TestCase):

	def test_find_bounding_box(self):
		'''
		Tests that correct non-background bounding_boxes saved into
		property_df
		'''
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask, max_col_num)
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
				dtype = colony_mask.dtype), max_col_num)
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
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask, max_col_num)
		test_flat_coords = \
			self.col_prop_finder._find_flat_coordinates(single_colony_mask)
		test_mask = np.zeros(single_colony_mask.shape,
			dtype = single_colony_mask.dtype)
		test_mask.flat[np.fromstring(test_flat_coords,
			dtype = int, sep = ' ')] = True
		assert_array_equal(single_colony_mask, test_mask)

class TestFindContourProps(unittest.TestCase):

	def test_find_contour_props(self):
		'''
		Tests that correct external colony perimeter and major axis
		length are returned
		'''
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask, max_col_num)
		expected_perimeter = 10.0
		expected_major_axis_length = 4.013865
		test_perimeter, test_major_axis_length = \
			self.col_prop_finder._find_contour_props(single_colony_mask)
		self.assertEqual(expected_perimeter, test_perimeter)
		assert_allclose(expected_major_axis_length, test_major_axis_length)

class TestFindColonyWiseProperties(unittest.TestCase):

	def test_find_colonywise_properties(self):
		'''
		Tests that correct perimeter and colony pixel index string
		identified
		'''
		self.col_prop_finder = _ColonyPropertyFinder(colony_mask, max_col_num)
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_colonywise_properties()
		# perimeter value is counterintuitive here but makes sense when
		# you realize contour tracks left to right first
		expected_property_df = \
			pd.DataFrame({
				'perimeter': [6+np.sqrt(2)+np.sqrt(2), 8+np.sqrt(2)],
				'pixel_idx_list': ['12 13 14 20 21 22 29 30 37',
					'25 26 33 34 35 41 43 49 50 51'],
				'major_axis_length': [3.4324963092803955, 3.7621912956237793]
				})
		assert_frame_equal(expected_property_df,
			self.col_prop_finder.property_df)

	def test_find_colonywise_properties_0(self):
		'''
		Tests that no centroids returned for empty colony mask
		'''
		pass
		self.col_prop_finder = \
			_ColonyPropertyFinder(np.zeros(colony_mask.shape,
				dtype = colony_mask.dtype), max_col_num)
		self.col_prop_finder._find_connected_components()
		self.col_prop_finder._find_centroids()
		expected_property_df = \
			pd.DataFrame({'cx': [], 'cy': []})
		#assert_frame_equal(expected_property_df,
		#	self.col_prop_finder.property_df)

class TestPixelIdxListtoMask(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.colony_property_finder_standin = \
			object.__new__(_ColonyPropertyFinder)
		self.colony_property_finder_standin.max_col_num = max_col_num

	def test_pixel_idx_list_to_mask(self):
		'''
		Tests conversion of pixel index list string to mask
		'''
		pixel_idx_list = '25 26 27 33 35 41 43 49 50 51'
		mask_shape = single_colony_mask.shape
		expected_colony_mask = single_colony_mask
		test_colony_mask = \
			self.colony_property_finder_standin._pixel_idx_list_to_mask(
				pixel_idx_list, mask_shape)
		assert_array_equal(expected_colony_mask, test_colony_mask)

	def test_pixel_idx_list_to_mask_blank(self):
		'''
		Tests conversion of empty pixel index list string to all-False
		mask
		'''
		pixel_idx_list = ''
		mask_shape = single_colony_mask.shape
		expected_colony_mask = np.zeros(mask_shape, dtype = bool)
		test_colony_mask = \
			self.colony_property_finder_standin._pixel_idx_list_to_mask(
				pixel_idx_list, mask_shape)
		assert_array_equal(expected_colony_mask, test_colony_mask)

class TestExpandBoundingBox(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.colony_property_finder_standin = \
			object.__new__(_ColonyPropertyFinder)
		self.colony_property_finder_standin.max_col_num = max_col_num

	def test_expand_bounding_box_default(self):
		'''
		Tests expanding bounding box by 5 pixels when it is not near
		an image edge
		'''
		bounding_box_series = \
			pd.Series({'bb_x_left': 10,
				'bb_y_top': 8,
				'bb_width': 4,
				'bb_height': 5}).astype('int32')
		im_row_num = 20
		im_col_num = 30
		expansion_pixels = 5
		# expected output vals
		y_start_exp = 3
		x_start_exp = 5
		y_range_end_exp = 18
		x_range_end_exp = 19
		# run the function
		y_start_test, x_start_test, y_range_end_test, x_range_end_test = \
			self.colony_property_finder_standin._expand_bounding_box(
				bounding_box_series, im_row_num, im_col_num, expansion_pixels)
		self.assertEqual(y_start_exp, y_start_test)
		self.assertEqual(x_start_exp, x_start_test)
		self.assertEqual(y_range_end_exp, y_range_end_test)
		self.assertEqual(x_range_end_exp, x_range_end_test)

	def test_expand_bounding_box_0_exp_val(self):
		'''
		Tests expanding bounding box by 0 pixels when it is not
		near an image edge
		'''
		bounding_box_series = \
			pd.Series({'bb_x_left': 10,
				'bb_y_top': 8,
				'bb_width': 4,
				'bb_height': 5}).astype('int32')
		im_row_num = 20
		im_col_num = 30
		expansion_pixels = 0
		# expected output vals
		y_start_exp = 8
		x_start_exp = 10
		y_range_end_exp = 13
		x_range_end_exp = 14
		# run the function
		y_start_test, x_start_test, y_range_end_test, x_range_end_test = \
			self.colony_property_finder_standin._expand_bounding_box(
				bounding_box_series, im_row_num, im_col_num, expansion_pixels)
		self.assertEqual(y_start_exp, y_start_test)
		self.assertEqual(x_start_exp, x_start_test)
		self.assertEqual(y_range_end_exp, y_range_end_test)
		self.assertEqual(x_range_end_exp, x_range_end_test)

	def test_expand_bounding_box_edge_overlap(self):
		'''
		Tests expanding bounding box by 5 pixels when expanded box would
		overlap an image edge
		'''
		bounding_box_series = \
			pd.Series({'bb_x_left': 4,
				'bb_y_top': 8,
				'bb_width': 10,
				'bb_height': 10}).astype('int32')
		im_row_num = 20
		im_col_num = 30
		expansion_pixels = 5
		# expected output vals
		y_start_exp = 3
		x_start_exp = 0
		y_range_end_exp = 20
		x_range_end_exp = 19
		# run the function
		y_start_test, x_start_test, y_range_end_test, x_range_end_test = \
			self.colony_property_finder_standin._expand_bounding_box(
				bounding_box_series, im_row_num, im_col_num, expansion_pixels)
		self.assertEqual(y_start_exp, y_start_test)
		self.assertEqual(x_start_exp, x_start_test)
		self.assertEqual(y_range_end_exp, y_range_end_test)
		self.assertEqual(x_range_end_exp, x_range_end_test)

class TestSubsetImbyBoundingBox(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.colony_property_finder_standin = \
			object.__new__(_ColonyPropertyFinder)
		self.colony_property_finder_standin.max_col_num = max_col_num

	def test_subset_im_by_bounding_box(self):
		'''
		Tests taking a subset of a numpy array based on bounding box and
		number of expansion_pixels
		'''
		im = colony_mask.astype(int)
		bounding_box_series = \
			pd.Series({'bb_x_left': 2,
				'bb_y_top': 1,
				'bb_width': 3,
				'bb_height': 5}).astype('int32')
		expansion_pixels = 1
		expected_im = np.array([
			[0, 0, 0, 0, 0],
			[0, 0, 0, 1, 1],
			[0, 0, 0, 1, 1],
			[1, 1, 0, 0, 1],
			[1, 1, 1, 0, 1],
			[1, 0, 1, 0, 0],
			[1, 1, 1, 0, 0]])
		test_im = \
			self.colony_property_finder_standin._subset_im_by_bounding_box(
				im, bounding_box_series, expansion_pixels)
		assert_array_equal(expected_im, test_im)

class TestGetErodedColonyMask(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.colony_mask_large = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
			[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
			[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
			[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
			[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype = bool)
		self.colony_property_finder = \
			_ColonyPropertyFinder(self.colony_mask_large, max_col_num)

	def test_get_eroded_colony_mask_expand_0(self):
		'''
		Tests getting an eroded colony mask using default structural
		element, with expansion_pixels set to 0
		'''
		bounding_box_series = \
			pd.Series({'bb_x_left': 1,
				'bb_y_top': 1,
				'bb_width': 10,
				'bb_height': 9}).astype('int32')
		expansion_pixels = 0
		expected_eroded_mask = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
			[0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
			[0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype = bool)
		test_eroded_mask = \
			self.colony_property_finder._get_eroded_bounded_mask(
				self.colony_mask_large, bounding_box_series, expansion_pixels)
		assert_array_equal(expected_eroded_mask, test_eroded_mask)

	def test_get_eroded_colony_mask_expand_3(self):
		'''
		Tests getting an eroded colony mask using default structural
		element, with expansion_pixels set to 3
		'''
		bounding_box_series = \
			pd.Series({'bb_x_left': 4,
				'bb_y_top': 4,
				'bb_width': 4,
				'bb_height': 3}).astype('int32')
		expansion_pixels = 3
		expected_eroded_mask = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
			[0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
			[0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype = bool)
		test_eroded_mask = \
			self.colony_property_finder._get_eroded_bounded_mask(
				self.colony_mask_large, bounding_box_series, expansion_pixels)
		assert_array_equal(expected_eroded_mask, test_eroded_mask)

class TestGetFilteredIntensities(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.colony_property_finder_standin = \
			object.__new__(_ColonyPropertyFinder)
		self.colony_property_finder_standin.max_col_num = max_col_num
		self.fluor_im = np.uint16(
			[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

	def test_get_filtered_intensities(self):
		'''
		Tests getting intensities from a numpy array based on bool mask
		with a fluorescence threshold
		'''
		mask = \
			np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
				dtype = bool)
		fluor_threshold = 12
		expected_mask_intensities_filtered = \
			np.uint16([1, 3, 6, 8, 9, 11])
		test_mask_intensities_filtered = \
			self.colony_property_finder_standin._get_filtered_intensities(
				self.fluor_im, mask, fluor_threshold)
		assert_array_equal(expected_mask_intensities_filtered,
			test_mask_intensities_filtered)

	def test_get_filtered_intensities_empty_mask(self):
		'''
		Tests getting intensities from a numpy array based on empty bool
		mask (so empty array returned)
		'''
		mask = \
			np.zeros(self.fluor_im.shape, dtype = bool)
		fluor_threshold = 12
		expected_mask_intensities_filtered = \
			np.uint16([])
		test_mask_intensities_filtered = \
			self.colony_property_finder_standin._get_filtered_intensities(
				self.fluor_im, mask, fluor_threshold)
		assert_array_equal(expected_mask_intensities_filtered,
			test_mask_intensities_filtered)

class TestMeasureColonyFluorProperties(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.colony_property_finder_standin = \
			object.__new__(_ColonyPropertyFinder)
		self.colony_property_finder_standin.max_col_num = max_col_num
		self.fluor_im = np.uint16([
			[1, 2, 3, 4, 5, 6],
			[7, 8, 9, 10, 11, 12],
			[13, 14, 15, 16, 17, 18],
			[19, 20, 21, 22, 23, 24],
			[25, 26, 27, 28, 29, 30],
			[31, 32, 33, 34, 35, 36]])
		self.single_colony_bbox_mask_eroded = np.array([
			[0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0],
			[0, 0, 1, 1, 0, 0],
			[0, 0, 1, 1, 0, 0],
			[0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0]], dtype = bool)
		self.background_bbox_mask_eroded = np.array([
			[1, 1, 1, 1, 1, 1],
			[1, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 1],
			[1, 0, 0, 0, 0, 1],
			[1, 1, 1, 1, 1, 1]], dtype = bool)

	def test_measure_colony_fluor_properties(self):
		'''
		Tests measuring background and colony properties of
		self.fluor_im without a fluorescent threshold
		'''
		fluor_threshold = np.inf
		colony_intensities = np.uint16([15, 16, 21, 22])
		background_intensities = \
			np.uint16([1, 2, 3, 4, 5, 6, 7, 12, 13, 18, 19, 24, 25, 30, 31, 32,
				33, 34, 35, 36])
		expected_fluor_prop_dict = dict()
		expected_fluor_prop_dict['back_mean_ppix_flprop'] = \
			np.mean(background_intensities)
		expected_fluor_prop_dict['back_med_ppix_flprop'] = \
			np.median(background_intensities)
		expected_fluor_prop_dict['back_var_ppix_flprop'] = \
			np.var(background_intensities)
		expected_fluor_prop_dict['col_mean_ppix_flprop'] = np.mean(colony_intensities)
		expected_fluor_prop_dict['col_med_ppix_flprop'] = np.median(colony_intensities)
		expected_fluor_prop_dict['col_var_ppix_flprop'] = np.var(colony_intensities)
		expected_fluor_prop_dict['col_upquartile_ppix_flprop'] = \
			np.quantile(colony_intensities, 0.75)
		test_fluor_prop_dict = \
			self.colony_property_finder_standin._measure_colony_fluor_properties(
				self.fluor_im, self.single_colony_bbox_mask_eroded,
				self.background_bbox_mask_eroded, fluor_threshold)
		self.assertDictEqual(expected_fluor_prop_dict, test_fluor_prop_dict)

	def test_measure_colony_fluor_properties_without_colony(self):
		'''
		Tests measuring background and colony properties of
		self.fluor_im, but setting a fluorescent threshold such that no
		colony data would be included
		'''
		fluor_threshold = 13
		background_intensities = \
			np.uint16([1, 2, 3, 4, 5, 6, 7, 12, 13])
		expected_fluor_prop_dict = dict()
		expected_fluor_prop_dict['back_mean_ppix_flprop'] = \
			np.mean(background_intensities)
		expected_fluor_prop_dict['back_med_ppix_flprop'] = \
			np.median(background_intensities)
		expected_fluor_prop_dict['back_var_ppix_flprop'] = \
			np.var(background_intensities)
		expected_fluor_prop_dict['col_mean_ppix_flprop'] = np.nan
		expected_fluor_prop_dict['col_med_ppix_flprop'] = np.nan
		expected_fluor_prop_dict['col_var_ppix_flprop'] = np.nan
		expected_fluor_prop_dict['col_upquartile_ppix_flprop'] = np.nan
		test_fluor_prop_dict = \
			self.colony_property_finder_standin._measure_colony_fluor_properties(
				self.fluor_im, self.single_colony_bbox_mask_eroded,
				self.background_bbox_mask_eroded, fluor_threshold)
		self.assertDictEqual(expected_fluor_prop_dict, test_fluor_prop_dict)

class TestSetUpFluorMeasurements(unittest.TestCase):

	def _read_saved_eroded_mask_ims(self, im_name):
		'''
		Reads saved IMs for colony mask and eroded colony and background
		masks
		'''
		colony_mask_path = os.path.join('PIE_tests', 'test_ims',
			(im_name + '_colony_mask.tif'))
		eroded_background_mask_path = os.path.join('PIE_tests', 'test_ims',
			(im_name + '_eroded_background_mask.tif'))
		eroded_colony_mask_path = os.path.join('PIE_tests', 'test_ims',
			(im_name + '_eroded_colony_mask.tif'))
		colony_mask = \
			cv2.imread(colony_mask_path, cv2.IMREAD_ANYDEPTH).astype(bool)
		eroded_background_mask = \
			cv2.imread(eroded_background_mask_path,
				cv2.IMREAD_ANYDEPTH).astype(bool)
		eroded_colony_mask = \
			cv2.imread(eroded_colony_mask_path,
				cv2.IMREAD_ANYDEPTH).astype(bool)
		return(colony_mask, eroded_colony_mask, eroded_background_mask)

	def test_setup_fluor_measurements_x0001(self):
		'''
		Tests setup of fluor measurement of small part of xy0001 from
		SL_170614_2_SC
		'''
		colony_mask, eroded_colony_mask, eroded_background_mask = \
			self._read_saved_eroded_mask_ims('xy0001c1_small')
		colony_property_finder = _ColonyPropertyFinder(colony_mask, max_col_num)
		colony_property_finder.measure_and_record_colony_properties()
		colony_property_finder.set_up_fluor_measurements()
		for _, row in colony_property_finder.property_df.iterrows():
			colony_bounding_box_series = \
				row[['bb_x_left','bb_y_top','bb_width','bb_height']]
			expected_ero_colony_mask = \
				colony_property_finder._subset_im_by_bounding_box(
					eroded_colony_mask, colony_bounding_box_series,
					colony_property_finder._fluor_measure_expansion_pixels)
			expected_ero_background_mask = \
				colony_property_finder._subset_im_by_bounding_box(
					eroded_background_mask, colony_bounding_box_series,
					colony_property_finder._fluor_measure_expansion_pixels)
			test_ero_colony_mask = row['Eroded_Colony_Mask']
			test_ero_background_mask = row['Eroded_Background_Mask']
			assert_array_equal(expected_ero_colony_mask, test_ero_colony_mask)
			assert_array_equal(expected_ero_background_mask,
				test_ero_background_mask)

	def test_setup_fluor_measurements_overlap_bb(self):
		'''
		Tests setup of fluor measurement of colonies with overlapping
		bounding boxes
		To do this, uses custom (small) dilation structural element and
		expansion pixel number
		'''
		colony_mask = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
			[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
			[0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
			[0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
			[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
			[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype = bool)
		expansion_pixels = 1
		colony_property_finder = \
			_ColonyPropertyFinder(colony_mask, max_col_num,
				fluor_measure_expansion_pixels = expansion_pixels)
		colony_property_finder._fluor_mask_erosion_kernel = \
			np.uint8([[0,1,0], [1,1,1], [0,1,0]])
		colony_property_finder.measure_and_record_colony_properties()
		colony_property_finder.set_up_fluor_measurements()
		expected_ero_colony_mask_0 = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 0, 0],
			[0, 0, 0, 0, 1, 1, 0, 0],
			[0, 0, 0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0]], dtype = bool)
		expected_ero_colony_mask_1 = np.array([
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1, 1, 0, 0, 0],
			[0, 0, 1, 1, 0, 0, 0],
			[0, 0, 0, 1, 1, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0]], dtype = bool)
		expected_ero_background_mask_0 = np.array([
			[1, 1, 1, 1, 0, 0, 1, 1],
			[1, 0, 0, 0, 0, 0, 0, 1],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 1]], dtype = bool)
		expected_ero_background_mask_1 = np.array([
			[1, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[1, 0, 0, 0, 0, 0, 0],
			[1, 1, 0, 0, 0, 0, 0],
			[1, 1, 1, 0, 0, 0, 1]], dtype = bool)
		test_ero_colony_mask_0 = \
			colony_property_finder.property_df.at[0, 'Eroded_Colony_Mask']
		test_ero_colony_mask_1 = \
			colony_property_finder.property_df.at[1, 'Eroded_Colony_Mask']
		test_ero_background_mask_0 = \
			colony_property_finder.property_df.at[0, 'Eroded_Background_Mask']
		test_ero_background_mask_1 = \
			colony_property_finder.property_df.at[1, 'Eroded_Background_Mask']
		assert_array_equal(expected_ero_colony_mask_0, test_ero_colony_mask_0)
		assert_array_equal(expected_ero_colony_mask_1, test_ero_colony_mask_1)
		assert_array_equal(expected_ero_background_mask_0,
			test_ero_background_mask_0)
		assert_array_equal(expected_ero_background_mask_1,
			test_ero_background_mask_1)

class TestMakeFluorMeasurements(unittest.TestCase):

	def test_make_fluor_measurements_x0001(self):
		'''
		Tests making fluorescence measurement of 2 colonies in a small
		part of xy0001 from SL_170614_2_SC, based on matlab results from
		pre-created eroded masks
		Notes:
		1. 	matlab code uses different way of calculating bounding box,
			potentially resulting in bigger bounding boxes; this was
			corrected for here
		2. 	matlab code uses values normalized to 2^bitdepth-1 (so max
			value of any pixel is 1); here, values not normalized by
			bitdepth
		3.	matlab seems to calculate quantile slightly differently; the
			one value affects by this has been corrected here

		'''
		# get colony mask
		colony_mask_path = os.path.join('PIE_tests', 'test_ims',
			'xy0001c1_small_colony_mask.tif')
		colony_mask = \
			cv2.imread(colony_mask_path, cv2.IMREAD_ANYDEPTH).astype(bool)
		# get fluorescent image
		fluor_im_path = \
			os.path.join('PIE_tests', 'test_ims', 'xy0001c2_small.tif')
		fluor_im = \
			cv2.imread(fluor_im_path, cv2.IMREAD_ANYDEPTH)
		# create _ColonyPropertyFinder object, set up for measurement
		colony_property_finder = _ColonyPropertyFinder(colony_mask, max_col_num)
		colony_property_finder.measure_and_record_colony_properties()
		colony_property_finder.set_up_fluor_measurements()
		# make fluorescent measurements
		fluor_channel_name = 'Green'
		fluor_threshold = np.inf
		colony_property_finder.make_fluor_measurements(fluor_im,
			fluor_channel_name, fluor_threshold)
		expected_fluor_prop_df = pd.DataFrame()
		expected_fluor_prop_df['back_mean_ppix_flprop_Green'] = \
			[131.4806201550386, 130.2075055187637]
		expected_fluor_prop_df['back_med_ppix_flprop_Green'] = \
			[130.5, 130.0]
		expected_fluor_prop_df['back_var_ppix_flprop_Green'] = \
			[88.737996514632457, 89.400650068954249]
		expected_fluor_prop_df['col_mean_ppix_flprop_Green'] = \
			[147.3103448275862, 131.2790697674419]
		expected_fluor_prop_df['col_med_ppix_flprop_Green'] = \
			[151.0, 132.0]
		expected_fluor_prop_df['col_var_ppix_flprop_Green'] = \
			[128.2829964328181, 95.2709572742023]
		expected_fluor_prop_df['col_upquartile_ppix_flprop_Green'] = \
			[155.0, 137.0]
		test_fluor_prop_df = \
			colony_property_finder.property_df[
				['back_mean_ppix_flprop_Green', 'back_med_ppix_flprop_Green',
				'back_var_ppix_flprop_Green', 'col_mean_ppix_flprop_Green',
				'col_med_ppix_flprop_Green', 'col_var_ppix_flprop_Green',
				'col_upquartile_ppix_flprop_Green']]
		assert_frame_equal(expected_fluor_prop_df, test_fluor_prop_df)
