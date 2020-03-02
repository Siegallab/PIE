#!/usr/bin/python

import unittest
import cv2
import numpy as np
from PIE.colony_edge_detect import _EdgeDetector
from numpy.testing import assert_array_equal

class TestGetPiePieces(unittest.TestCase):

	edge_detector_standin = object.__new__(_EdgeDetector)
	edge_detector_standin.input_im = \
		cv2.imread('PIE_tests/test_ims/test_im_small.tif',
			cv2.IMREAD_ANYDEPTH)
	pie_quadrants = ['i', 'ii', 'iii', 'iv']
	expected_pie_piece_dict = dict()
	for current_quadrant in pie_quadrants:
		expected_pie_piece_dict[current_quadrant] = \
			cv2.imread('PIE_tests/test_ims/test_im_small_pie_pieces_' +
				current_quadrant + '.tif', cv2.IMREAD_ANYDEPTH).astype(bool)

	def test_get_pie_pieces(self):
		'''
		Tests that PIE quadrants correctly identifed for input image
		'''
		self.edge_detector_standin._get_pie_pieces()
		for quad in self.pie_quadrants:
			assert_array_equal(self.expected_pie_piece_dict[quad],
				self.edge_detector_standin.pie_piece_dict[quad].pie_mask)

class TestFillHoles(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.edge_detector_standin = object.__new__(_EdgeDetector)
		self.input_mask = np.array([
			[False, False, False,  False, True,  True,  True],
			[False, False, False,  False, True,  False,  True],
			[ True,  True,  False, False, True,  True,  True],
			[ True,  True,  True,  True, False,  False, False],
			[ True,  False,  False,  True, True, False, False],
			[ True,  True,  True, True, False, False, False]])

	def test_hole_size_0(self):
		'''
		Tests that no holes filled when hole_fill_area is 0
		'''
		self.edge_detector_standin.hole_fill_area = 0
		expected_filled_holes_mask = self.input_mask
		test_filled_holes_mask = \
			self.edge_detector_standin._fill_holes(self.input_mask)
		assert_array_equal(expected_filled_holes_mask, test_filled_holes_mask)

	def test_hole_size_1(self):
		'''
		Tests that only small holes filled when hole_fill_area is 1
		'''
		self.edge_detector_standin.hole_fill_area = 1
		expected_filled_holes_mask = np.array([
			[False, False, False,  False, True,  True,  True],
			[False, False, False,  False, True,  True,  True],
			[ True,  True,  False, False, True,  True,  True],
			[ True,  True,  True,  True, False,  False, False],
			[ True,  False,  False,  True, True, False, False],
			[ True,  True,  True, True, False, False, False]])
		test_filled_holes_mask = \
			self.edge_detector_standin._fill_holes(self.input_mask)
		assert_array_equal(expected_filled_holes_mask, test_filled_holes_mask)

	def test_hole_size_inf(self):
		'''
		Tests that no holes filled when hole_fill_area is np.inf
		'''
		self.edge_detector_standin.hole_fill_area = np.inf
		expected_filled_holes_mask = np.array([
			[False, False, False,  False, True,  True,  True],
			[False, False, False,  False, True,  True,  True],
			[ True,  True,  False, False, True,  True,  True],
			[ True,  True,  True,  True, False,  False, False],
			[ True,  True, True,  True, True, False, False],
			[ True,  True,  True, True, False, False, False]])
		test_filled_holes_mask = \
			self.edge_detector_standin._fill_holes(self.input_mask)
		assert_array_equal(expected_filled_holes_mask, test_filled_holes_mask)

class TestClearMaskEdges(unittest.TestCase):

	@classmethod
	def setUp(self):
		self.edge_detector_standin = object.__new__(_EdgeDetector)

	def test_clear_nothing(self):
		'''
		Tests behavior on a mask with no objects touching image edge
		'''
		colony_mask = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 0],
			[0, 0, 0, 0, 1, 1, 1, 0],
			[0, 1, 1, 0, 1, 1, 1, 0],
			[0, 1, 0, 1, 0, 1, 0, 0],
			[0, 1, 0, 1, 0, 0, 0, 0],
			[0, 1, 1, 1, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0]], dtype = bool)
		test_cleared_mask = \
			self.edge_detector_standin._clear_mask_edges(colony_mask)
		assert_array_equal(colony_mask, test_cleared_mask)

	def test_clear_single_object(self):
		'''
		Tests behavior on a mask with one object touching image edge
		'''
		colony_mask = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 1],
			[0, 0, 0, 0, 1, 1, 1, 0],
			[0, 1, 1, 0, 1, 1, 1, 0],
			[0, 1, 0, 1, 0, 1, 0, 0],
			[0, 1, 0, 1, 0, 0, 0, 0],
			[0, 1, 1, 1, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0]], dtype = bool)
		expected_cleared_mask = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 1, 0, 0, 0, 0, 0],
			[0, 1, 0, 1, 0, 0, 0, 0],
			[0, 1, 0, 1, 0, 0, 0, 0],
			[0, 1, 1, 1, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0]], dtype = bool)
		test_cleared_mask = \
			self.edge_detector_standin._clear_mask_edges(colony_mask)
		assert_array_equal(expected_cleared_mask, test_cleared_mask)

	def test_clear_everything(self):
		'''
		Tests behavior on a mask with both objects touching image edge
		'''
		colony_mask = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 1],
			[0, 0, 0, 0, 1, 1, 1, 0],
			[0, 1, 1, 0, 1, 1, 1, 0],
			[0, 1, 0, 1, 0, 1, 0, 0],
			[0, 1, 0, 1, 0, 0, 0, 0],
			[0, 1, 1, 1, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 0, 0]], dtype = bool)
		expected_cleared_mask = np.zeros(colony_mask.shape, colony_mask.dtype)
		test_cleared_mask = \
			self.edge_detector_standin._clear_mask_edges(colony_mask)
		assert_array_equal(expected_cleared_mask, test_cleared_mask)

class TestClearMaskEdges(unittest.TestCase):

	# test this on EP_160110_t02xy1005_small.tif
	pass


if __name__ == '__main__':
	unittest.main()