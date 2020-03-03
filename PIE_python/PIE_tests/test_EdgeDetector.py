#!/usr/bin/python

import unittest
import cv2
import numpy as np
import os
from PIE.colony_edge_detect import _EdgeDetector, create_color_overlay
from numpy.testing import assert_array_equal
from time import sleep

def show_mask_diffs(green_mask, mag_mask, im, im_name):
	'''
	Displays im with pixels unique to green_mask as green and pixels
	unique to mag_mask as magenta
	'''
	norm_im = \
		cv2.normalize(im, None, alpha=0, beta=(2**8-1),
			norm_type=cv2.NORM_MINMAX)
	green_pixels = np.copy(green_mask)
	green_pixels[mag_mask] = False
	mag_pixels = np.copy(mag_mask)
	mag_pixels[green_mask] = False
	green_im = create_color_overlay(norm_im, green_pixels, [0, 255, 0], 1)
	green_mag_im = create_color_overlay(green_im, mag_pixels, [255, 0, 255], 1)
	out_path = os.path.join('PIE_tests', 'test_ims',
		('test_overlap_' + im_name + '.tif'))
	cv2.imwrite(out_path, green_mag_im * 2**8)

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


class TestCreateInitialColonyMask(unittest.TestCase):

	def _compare_overlays(self, im_name):
		hole_fill_area = 0
		# read in images
		im_path = os.path.join('PIE_tests', 'test_ims', (im_name + '.tif'))
		center_path = os.path.join('PIE_tests', 'test_ims',
			(im_name + '_cell_centers.tif'))
		initial_overlay_path = os.path.join('PIE_tests', 'test_ims',
			(im_name + '_initial_overlay.tif'))
		input_im = cv2.imread(im_path, cv2.IMREAD_ANYDEPTH)
		cell_centers = cv2.imread(center_path, cv2.IMREAD_ANYDEPTH).astype(bool)
		expected_initial_colony_mask = \
			cv2.imread(initial_overlay_path, cv2.IMREAD_ANYDEPTH).astype(bool)
		# run initial overlay identification
		edge_detector = _EdgeDetector(input_im, cell_centers, hole_fill_area,
			False, 0.25)
		test_initial_colony_mask = edge_detector._create_inital_colony_mask()
		if not np.array_equal(expected_initial_colony_mask,
			test_initial_colony_mask):
			show_mask_diffs(expected_initial_colony_mask,
				test_initial_colony_mask, input_im, im_name)
		assert_array_equal(expected_initial_colony_mask,
			test_initial_colony_mask)

	def test_test_im_small(self):
		'''
		Test initial overlay creation on test_im_small
		'''
		self._compare_overlays('test_im_small')

	def test_EP_160110_t02xy1005_small(self):
		'''
		Test initial overlay creation on EP_160110_t02xy1005_small
		'''
		self._compare_overlays('EP_160110_t02xy1005_small')

class TestRunEdgeDetection(unittest.TestCase):

	def _compare_overlays(self, im_name, cleanup):
		hole_fill_area = np.inf
		# read in images
		im_path = os.path.join('PIE_tests', 'test_ims', (im_name + '.tif'))
		center_path = os.path.join('PIE_tests', 'test_ims',
			(im_name + '_cell_centers.tif'))
		if cleanup:
			colony_mask_path = os.path.join('PIE_tests', 'test_ims',
				(im_name + '_colony_mask_cleanup.tif'))
		else:
			colony_mask_path = os.path.join('PIE_tests', 'test_ims',
				(im_name + '_colony_mask.tif'))
		input_im = cv2.imread(im_path, cv2.IMREAD_ANYDEPTH)
		cell_centers = cv2.imread(center_path, cv2.IMREAD_ANYDEPTH).astype(bool)
		expected_final_colony_mask = \
			cv2.imread(colony_mask_path, cv2.IMREAD_ANYDEPTH).astype(bool)
		# run initial overlay identification
		edge_detector = _EdgeDetector(input_im, cell_centers, hole_fill_area, cleanup, 0.25)
		test_final_colony_mask = edge_detector.run_edge_detection()
		if not np.array_equal(expected_final_colony_mask,
			test_final_colony_mask):
			show_mask_diffs(expected_final_colony_mask,
				test_final_colony_mask, input_im, im_name)
		assert_array_equal(expected_final_colony_mask,
			test_final_colony_mask)

	def test_test_im_small_no_cleanup(self):
		'''
		Test initial overlay creation on test_im_small
		'''
		self._compare_overlays('test_im_small', False)

	def test_EP_160110_t02xy1005_small_no_cleanup(self):
		'''
		Test initial overlay creation on EP_160110_t02xy1005_small
		Actually, original colony mask differs from matlab results by 8
		pixels that have been deleted, seemingly as a result of matlab
		including a 'valley' in image intensity in PIE pieces, and
		python not including it
		'''
		self._compare_overlays('EP_160110_t02xy1005_small', False)


if __name__ == '__main__':
	unittest.main()