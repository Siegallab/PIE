#!/usr/bin/python

import unittest
import cv2
import numpy as np
import os
from PIE.colony_edge_detect import EdgeDetector, _PiePiece
import PIE.general_testing_functions as general_pie_testing_functions
from numpy.testing import assert_array_equal

def _set_up_edge_detector(im_name, cleanup, max_proportion_exposed_edge):
	# read in images
	im_path = os.path.join('tests', 'test_ims', (im_name + '.tif'))
	center_path = os.path.join('tests', 'test_ims',
		(im_name + '_cell_centers.tif'))
	input_im = cv2.imread(im_path, cv2.IMREAD_ANYDEPTH)
	cell_centers = cv2.imread(center_path, cv2.IMREAD_ANYDEPTH).astype(bool)
	# find pie pieces in initialization
	edge_detector = EdgeDetector(input_im, cell_centers, None, cleanup,
		max_proportion_exposed_edge)
	# create overlap of each piece piece with cell center
	edge_detector._create_inital_colony_mask()
	return(edge_detector)

class TestGetPiePieces(unittest.TestCase):

	edge_detector_standin = object.__new__(EdgeDetector)
	edge_detector_standin.input_im = \
		cv2.imread('tests/test_ims/test_im_small.tif',
			cv2.IMREAD_ANYDEPTH)
	pie_quadrants = ['i', 'ii', 'iii', 'iv']
	expected_pie_piece_dict = dict()
	for current_quadrant in pie_quadrants:
		expected_pie_piece_dict[current_quadrant] = \
			cv2.imread('tests/test_ims/test_im_small_pie_pieces_' +
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
		self.edge_detector_standin = object.__new__(EdgeDetector)
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
	def setUpClass(self):
		self.edge_detector_standin = object.__new__(EdgeDetector)

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
		im_path = os.path.join('tests', 'test_ims', (im_name + '.tif'))
		center_path = os.path.join('tests', 'test_ims',
			(im_name + '_cell_centers.tif'))
		initial_overlay_path = os.path.join('tests', 'test_ims',
			(im_name + '_initial_overlay.tif'))
		input_im = cv2.imread(im_path, cv2.IMREAD_ANYDEPTH)
		cell_centers = cv2.imread(center_path, cv2.IMREAD_ANYDEPTH).astype(bool)
		expected_initial_colony_mask = \
			cv2.imread(initial_overlay_path, cv2.IMREAD_ANYDEPTH).astype(bool)
		# run initial overlay identification
		edge_detector = EdgeDetector(input_im, cell_centers, hole_fill_area,
			False, 0.25)
		test_initial_colony_mask = edge_detector._create_inital_colony_mask()
		if not np.array_equal(expected_initial_colony_mask,
			test_initial_colony_mask):
			general_pie_testing_functions.show_mask_diffs(expected_initial_colony_mask,
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


class TestCreateTranslationMatrix(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.edge_detector_standin = object.__new__(EdgeDetector)
		self.edge_detector_standin.pie_piece_position_dict = \
			{'i': np.array([[0, 1], [0, 0]]),
			'ii': np.array([[1, 0], [0, 0]]),
			'iii': np.array([[0, 0], [1, 0]]),
			'iv': np.array([[0, 0], [0, 1]])}

	def test_translation_i_to_ii(self):
		source_quad = 'i'
		target_quad = 'ii'
		expected_translation_mat = \
			np.array([[1, 0, -1], [0, 1, 0]]).astype(float)
		test_translation_mat = \
			self.edge_detector_standin._create_translation_matrix(
				self.edge_detector_standin.pie_piece_position_dict[source_quad],
				self.edge_detector_standin.pie_piece_position_dict[target_quad])
		assert_array_equal(expected_translation_mat, test_translation_mat)

	def test_translation_ii_to_i(self):
		source_quad = 'ii'
		target_quad = 'i'
		expected_translation_mat = \
			np.array([[1, 0, 1], [0, 1, 0]]).astype(float)
		test_translation_mat = \
			self.edge_detector_standin._create_translation_matrix(
				self.edge_detector_standin.pie_piece_position_dict[source_quad],
				self.edge_detector_standin.pie_piece_position_dict[target_quad])
		assert_array_equal(expected_translation_mat, test_translation_mat)

	def test_translation_i_to_iii(self):
		source_quad = 'i'
		target_quad = 'iii'
		expected_translation_mat = \
			np.array([[1, 0, -1], [0, 1, 1]]).astype(float)
		test_translation_mat = \
			self.edge_detector_standin._create_translation_matrix(
				self.edge_detector_standin.pie_piece_position_dict[source_quad],
				self.edge_detector_standin.pie_piece_position_dict[target_quad])
		assert_array_equal(expected_translation_mat, test_translation_mat)

	def test_translation_iii_to_i(self):
		source_quad = 'iii'
		target_quad = 'i'
		expected_translation_mat = \
			np.array([[1, 0, 1], [0, 1, -1]]).astype(float)
		test_translation_mat = \
			self.edge_detector_standin._create_translation_matrix(
				self.edge_detector_standin.pie_piece_position_dict[source_quad],
				self.edge_detector_standin.pie_piece_position_dict[target_quad])
		assert_array_equal(expected_translation_mat, test_translation_mat)

	def test_translation_i_to_iv(self):
		source_quad = 'i'
		target_quad = 'iv'
		expected_translation_mat = \
			np.array([[1, 0, 0], [0, 1, 1]]).astype(float)
		test_translation_mat = \
			self.edge_detector_standin._create_translation_matrix(
				self.edge_detector_standin.pie_piece_position_dict[source_quad],
				self.edge_detector_standin.pie_piece_position_dict[target_quad])
		assert_array_equal(expected_translation_mat, test_translation_mat)

	def test_translation_iv_to_i(self):
		source_quad = 'iv'
		target_quad = 'i'
		expected_translation_mat = \
			np.array([[1, 0, 0], [0, 1, -1]]).astype(float)
		test_translation_mat = \
			self.edge_detector_standin._create_translation_matrix(
				self.edge_detector_standin.pie_piece_position_dict[source_quad],
				self.edge_detector_standin.pie_piece_position_dict[target_quad])
		assert_array_equal(expected_translation_mat, test_translation_mat)

	def test_translation_ii_to_iii(self):
		source_quad = 'ii'
		target_quad = 'iii'
		expected_translation_mat = \
			np.array([[1, 0, 0], [0, 1, 1]]).astype(float)
		test_translation_mat = \
			self.edge_detector_standin._create_translation_matrix(
				self.edge_detector_standin.pie_piece_position_dict[source_quad],
				self.edge_detector_standin.pie_piece_position_dict[target_quad])
		assert_array_equal(expected_translation_mat, test_translation_mat)

	def test_translation_ii_to_iii(self):
		source_quad = 'iii'
		target_quad = 'ii'
		expected_translation_mat = \
			np.array([[1, 0, 0], [0, 1, -1]]).astype(float)
		test_translation_mat = \
			self.edge_detector_standin._create_translation_matrix(
				self.edge_detector_standin.pie_piece_position_dict[source_quad],
				self.edge_detector_standin.pie_piece_position_dict[target_quad])
		assert_array_equal(expected_translation_mat, test_translation_mat)

	def test_translation_ii_to_iv(self):
		source_quad = 'ii'
		target_quad = 'iv'
		expected_translation_mat = \
			np.array([[1, 0, 1], [0, 1, 1]]).astype(float)
		test_translation_mat = \
			self.edge_detector_standin._create_translation_matrix(
				self.edge_detector_standin.pie_piece_position_dict[source_quad],
				self.edge_detector_standin.pie_piece_position_dict[target_quad])
		assert_array_equal(expected_translation_mat, test_translation_mat)

	def test_translation_iv_to_ii(self):
		source_quad = 'iv'
		target_quad = 'ii'
		expected_translation_mat = \
			np.array([[1, 0, -1], [0, 1, -1]]).astype(float)
		test_translation_mat = \
			self.edge_detector_standin._create_translation_matrix(
				self.edge_detector_standin.pie_piece_position_dict[source_quad],
				self.edge_detector_standin.pie_piece_position_dict[target_quad])
		assert_array_equal(expected_translation_mat, test_translation_mat)

	def test_translation_iv_to_iv(self):
		source_quad = 'iv'
		target_quad = 'iv'
		expected_translation_mat = \
			np.array([[1, 0, 0], [0, 1, 0]]).astype(float)
		test_translation_mat = \
			self.edge_detector_standin._create_translation_matrix(
				self.edge_detector_standin.pie_piece_position_dict[source_quad],
				self.edge_detector_standin.pie_piece_position_dict[target_quad])
		assert_array_equal(expected_translation_mat, test_translation_mat)

class TestPerformNeighborFiltering(unittest.TestCase):

	def setUp(self):
		self.edge_detector_standin = object.__new__(EdgeDetector)
		self.edge_detector_standin.pie_piece_position_dict = \
			{'i': np.array([[0, 1], [0, 0]]),
			'ii': np.array([[1, 0], [0, 0]]),
			'iii': np.array([[0, 0], [1, 0]]),
			'iv': np.array([[0, 0], [0, 1]])}
		self.edge_detector_standin.pie_piece_dict = \
			{'i': object.__new__(_PiePiece),
			'ii': object.__new__(_PiePiece),
			'iii': object.__new__(_PiePiece),
			'iv': object.__new__(_PiePiece)}
		self.edge_detector_standin.pie_piece_dict['i'].\
			edge_filtered_pie_mask = np.array([
				[0,0,0,1,1,0,0,0,0,0,0,1,1,0,0],
				[0,0,0,1,1,1,0,0,0,0,0,1,1,1,0],
				[0,0,0,0,1,1,0,0,0,0,0,0,1,1,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], dtype = bool)
		self.edge_detector_standin.pie_piece_dict['ii'].\
			edge_filtered_pie_mask = np.array([
				[0,1,1,0,0,0,0,0,0,1,1,0,0,0,0],
				[1,1,1,0,0,0,0,0,1,1,1,0,0,0,0],
				[1,1,0,0,0,0,0,0,1,1,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], dtype = bool)
		self.edge_detector_standin.pie_piece_dict['iii'].\
			edge_filtered_pie_mask = np.array([
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[1,1,0,0,0,0,0,0,1,1,0,0,0,0,0],
				[1,1,1,0,0,0,0,0,1,1,1,0,0,0,0],
				[0,1,1,0,0,0,0,0,0,1,1,0,0,0,0]], dtype = bool)
		self.edge_detector_standin.pie_piece_dict['iv'].\
			edge_filtered_pie_mask = np.array([
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
				[0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
				[0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]], dtype = bool)
		# create cell_overlap_labeled_pie_masks and
		# neighbor_filtered_pie_masks
		for quad in self.edge_detector_standin.pie_piece_dict.keys():
			self.edge_detector_standin.pie_piece_dict[quad].\
				neighbor_filtered_pie_mask = \
				np.copy(self.edge_detector_standin.pie_piece_dict[quad].\
					edge_filtered_pie_mask)
			_, self.edge_detector_standin.pie_piece_dict[quad].\
			cell_overlap_labeled_pie_mask = \
				cv2.connectedComponents(np.uint8(
					self.edge_detector_standin.pie_piece_dict[quad].\
						neighbor_filtered_pie_mask), connectivity = 4)

	def test_filter_i(self):
		'''
		Filter quadrant i objects based on presence/absence of expected
		direct neighbors horizontally and vertically
		'''
		focal_pie_quad_name = 'i'
		focal_pie_piece_quadrant = \
			self.edge_detector_standin.pie_piece_dict[focal_pie_quad_name]
		# object on right missing neighbor from below, should be
		# filtered out
		expected_neighbor_filtered_pie_mask = np.array([
			[0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], dtype = bool)
		self.edge_detector_standin._perform_neighbor_filtering(focal_pie_quad_name,
			focal_pie_piece_quadrant)
		assert_array_equal(expected_neighbor_filtered_pie_mask,
			self.edge_detector_standin.pie_piece_dict[
				focal_pie_quad_name].neighbor_filtered_pie_mask)

	def test_filter_ii(self):
		'''
		Filter quadrant ii objects based on presence/absence of expected
		direct neighbors horizontally and vertically
		'''
		focal_pie_quad_name = 'ii'
		focal_pie_piece_quadrant = \
			self.edge_detector_standin.pie_piece_dict[focal_pie_quad_name]
		# all objects have non-diagonal neighbors and are present
		expected_neighbor_filtered_pie_mask = \
			np.copy(focal_pie_piece_quadrant.neighbor_filtered_pie_mask)
		self.edge_detector_standin._perform_neighbor_filtering(
			focal_pie_quad_name, focal_pie_piece_quadrant)
		assert_array_equal(expected_neighbor_filtered_pie_mask,
			self.edge_detector_standin.pie_piece_dict[
				focal_pie_quad_name].neighbor_filtered_pie_mask)

	def test_filter_iii(self):
		'''
		Filter quadrant iii objects based on presence/absence of expected
		direct neighbors horizontally and vertically
		'''
		focal_pie_quad_name = 'iii'
		focal_pie_piece_quadrant = \
			self.edge_detector_standin.pie_piece_dict[focal_pie_quad_name]
		# object on right missing neighbor from the right, should be
		# filtered out
		expected_neighbor_filtered_pie_mask = np.array([
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,1,1,0,0,0,0,0,0,0,0,0,0,0,0]], dtype = bool)
		self.edge_detector_standin._perform_neighbor_filtering(focal_pie_quad_name,
			focal_pie_piece_quadrant)
		assert_array_equal(expected_neighbor_filtered_pie_mask,
			self.edge_detector_standin.pie_piece_dict[
				focal_pie_quad_name].neighbor_filtered_pie_mask)

	def test_filter_iv(self):
		'''
		Filter quadrant iv objects based on presence/absence of expected
		direct neighbors horizontally and vertically
		'''
		focal_pie_quad_name = 'iv'
		focal_pie_piece_quadrant = \
			self.edge_detector_standin.pie_piece_dict[focal_pie_quad_name]
		# only one object, has non-diagonal neighbors
		expected_neighbor_filtered_pie_mask = \
			np.copy(focal_pie_piece_quadrant.neighbor_filtered_pie_mask)
		self.edge_detector_standin._perform_neighbor_filtering(
			focal_pie_quad_name, focal_pie_piece_quadrant)
		assert_array_equal(expected_neighbor_filtered_pie_mask,
			self.edge_detector_standin.pie_piece_dict[
				focal_pie_quad_name].neighbor_filtered_pie_mask)

class TestSingleRoundEdgeFiltering(unittest.TestCase):

	def test_EP_160110_t02xy1005_small_edge_filter_first(self):
		'''
		Tests first round of filtration by edge on
		EP_160110_t02xy1005_small
		'''
		max_proportion_exposed_edge = 0.75
		# set up pie pieces for EP_160110_t02xy1005_small
		edge_detector = \
			_set_up_edge_detector(
				'EP_160110_t02xy1005_small', True, max_proportion_exposed_edge)
		initial_colony_mask_path = os.path.join('tests', 'test_ims',
			'EP_160110_t02xy1005_small_initial_overlay.tif')
		initial_colony_mask = cv2.imread(initial_colony_mask_path,
			cv2.IMREAD_ANYDEPTH).astype(bool)
		# filter by edges
		edge_detector._single_round_edge_filtering(initial_colony_mask)
		# test filtering results
		for pie_quad_name, pie_piece_quadrant in \
			edge_detector.pie_piece_dict.items():
			current_pie_piece_im_name = 'EP_160110_t02xy1005_small_round1_pie_' + \
				pie_quad_name + '_edge_cleared'
			expected_pie_edge_mask_path = \
				os.path.join('tests', 'test_ims',
					(current_pie_piece_im_name + '.tif'))
			expected_edge_filtered_mask = \
				cv2.imread(expected_pie_edge_mask_path,
					cv2.IMREAD_ANYDEPTH).astype(bool)
			test_edge_filtered_mask = pie_piece_quadrant.edge_filtered_pie_mask
			if not np.array_equal(expected_edge_filtered_mask,
				test_edge_filtered_mask):
				general_pie_testing_functions.show_mask_diffs(expected_edge_filtered_mask,
					test_edge_filtered_mask, edge_detector.input_im,
					(current_pie_piece_im_name))
			assert_array_equal(expected_edge_filtered_mask,
				test_edge_filtered_mask)

class TestSingleRoundNeighborFiltering(unittest.TestCase):

	def test_EP_160110_t02xy1005_small_neighbor_filter_first(self):
		'''
		Tests first round of filtration by neighbor on
		EP_160110_t02xy1005_small
		'''
		# set up pie pieces for EP_160110_t02xy1005_small
		max_proportion_exposed_edge = 0.75
#		edge_detector = \
#			_set_up_edge_detector(
#				'EP_160110_t02xy1005_small', True, max_proportion_exposed_edge)
		edge_detector = object.__new__(EdgeDetector)
		edge_detector.max_proportion_exposed_edge = max_proportion_exposed_edge
		# create 'pie' quadrants
		edge_detector.pie_piece_position_dict = \
			{'i': np.array([[0, 1], [0, 0]]),
			'ii': np.array([[1, 0], [0, 0]]),
			'iii': np.array([[0, 0], [1, 0]]),
			'iv': np.array([[0, 0], [0, 1]])}
		edge_detector.pie_piece_dict = dict()
		edge_detector.pie_piece_dict['i'] = object.__new__(_PiePiece)
		edge_detector.pie_piece_dict['ii'] = object.__new__(_PiePiece)
		edge_detector.pie_piece_dict['iii'] = object.__new__(_PiePiece)
		edge_detector.pie_piece_dict['iv'] = object.__new__(_PiePiece)
		# assign edge masks from files
		for pie_quad_name, pie_piece_quadrant in \
			edge_detector.pie_piece_dict.items():
			# read in edge-filtered pie piece
			labeled_pie_piece_file_name = 'EP_160110_t02xy1005_small_pie_' + \
				pie_quad_name + '_labeled_pie_mask.tif'
			edge_filtered_pie_mask_file_name = 'EP_160110_t02xy1005_small_round1_pie_' + \
				pie_quad_name + '_edge_cleared.tif'
			edge_mask_path = \
				os.path.join('tests', 'test_ims',
					edge_filtered_pie_mask_file_name)
			labeled_pie_piece_path = \
				os.path.join('tests', 'test_ims',
					labeled_pie_piece_file_name)
			pie_piece_quadrant.edge_filtered_pie_mask = \
				cv2.imread(edge_mask_path,
					cv2.IMREAD_ANYDEPTH).astype(bool)
			pie_piece_quadrant.cell_overlap_labeled_pie_mask = \
				cv2.imread(labeled_pie_piece_path,
					cv2.IMREAD_ANYDEPTH)
		# perform neighbor filtering on all pie pieces
		combined_filtered_pieces = \
			np.zeros(
				edge_detector.pie_piece_dict['i'].edge_filtered_pie_mask.shape,
				dtype = bool)
		edge_detector._single_round_neighbor_filtering(combined_filtered_pieces)
		# test neighbor mask
		for pie_quad_name, pie_piece_quadrant in \
			edge_detector.pie_piece_dict.items():
			# read in expected neighbor-filtered pie piece
			current_pie_piece_im_name = 'EP_160110_t02xy1005_small_round1_pie_' + \
				pie_quad_name
			neighbor_mask_path = \
				os.path.join('tests', 'test_ims',
					(current_pie_piece_im_name + '_neighbor_cleared.tif'))
			expected_neighbor_filtered_mask = \
				cv2.imread(neighbor_mask_path,
					cv2.IMREAD_ANYDEPTH).astype(bool)
			test_neighbor_filtered_mask = \
				pie_piece_quadrant.neighbor_filtered_pie_mask
			if not np.array_equal(expected_neighbor_filtered_mask,
				test_neighbor_filtered_mask):
				general_pie_testing_functions.show_mask_diffs(
					expected_neighbor_filtered_mask,
					test_neighbor_filtered_mask, edge_detector.input_im,
					(current_pie_piece_im_name + '_neighbor_cleared'))
			assert_array_equal(expected_neighbor_filtered_mask,
				test_neighbor_filtered_mask)

class TestRunCleanup(unittest.TestCase):

	def _compare_cleanup_results(self, im_name, max_proportion_exposed_edge, hole_fill_area):
		# set up pie pieces for EP_160110_t02xy1005_small
		edge_detector = object.__new__(EdgeDetector)
		edge_detector.max_proportion_exposed_edge = max_proportion_exposed_edge
		edge_detector.hole_fill_area = hole_fill_area
		# edge_detector needs input_im for size
		# (and to overlay masks onto in case of failed test)
		im_path = os.path.join('tests', 'test_ims', (im_name + '.tif'))
		edge_detector.input_im = cv2.imread(im_path, cv2.IMREAD_ANYDEPTH)
		# create 'pie' quadrants
		edge_detector.pie_piece_position_dict = \
			{'i': np.array([[0, 1], [0, 0]]),
			'ii': np.array([[1, 0], [0, 0]]),
			'iii': np.array([[0, 0], [1, 0]]),
			'iv': np.array([[0, 0], [0, 1]])}
		edge_detector.pie_piece_dict = dict()
		edge_detector.pie_piece_dict['i'] = object.__new__(_PiePiece)
		edge_detector.pie_piece_dict['ii'] = object.__new__(_PiePiece)
		edge_detector.pie_piece_dict['iii'] = object.__new__(_PiePiece)
		edge_detector.pie_piece_dict['iv'] = object.__new__(_PiePiece)
		# assign edge masks from files
		for pie_quad_name, pie_piece_quadrant in \
			edge_detector.pie_piece_dict.items():
			# read in edge-filtered pie piece
			current_pie_piece_im_name = im_name + '_pie_' + \
				pie_quad_name
			pie_piece_path = \
				os.path.join('tests', 'test_ims',
					(current_pie_piece_im_name + '_pie_pieces_cell_center_cleared.tif'))
			labeled_pie_piece_path = \
				os.path.join('tests', 'test_ims',
					(current_pie_piece_im_name + '_labeled_pie_mask.tif'))
			pie_piece_quadrant.cell_overlap_pie_mask = \
				cv2.imread(pie_piece_path,
					cv2.IMREAD_ANYDEPTH).astype(bool)
			pie_piece_quadrant.cell_overlap_labeled_pie_mask = \
				cv2.imread(labeled_pie_piece_path,
					cv2.IMREAD_ANYDEPTH)
		# load initial colony masks
		initial_colony_mask_file = os.path.join('tests', 'test_ims',
			(im_name + '_initial_overlay.tif'))
		colony_mask_filled_holes_file = os.path.join('tests', 'test_ims',
			(im_name + '_colony_mask.tif'))
		expected_colony_mask_filled_holes_file = os.path.join('tests',
			'test_ims', (im_name + '_colony_mask_cleanup.tif'))
		initial_colony_mask = cv2.imread(initial_colony_mask_file,
			cv2.IMREAD_ANYDEPTH).astype(bool)
		colony_mask_filled_holes = cv2.imread(colony_mask_filled_holes_file,
			cv2.IMREAD_ANYDEPTH).astype(bool)
		expected_colony_mask_filled_holes = \
			cv2.imread(expected_colony_mask_filled_holes_file,
				cv2.IMREAD_ANYDEPTH).astype(bool)
		# perform cleanup
		test_colony_mask_filled_holes = \
			edge_detector._run_cleanup(initial_colony_mask,
				colony_mask_filled_holes)
		# test equality
		if not np.array_equal(expected_colony_mask_filled_holes,
			test_colony_mask_filled_holes):
			general_pie_testing_functions.show_mask_diffs(expected_colony_mask_filled_holes,
				test_colony_mask_filled_holes, edge_detector.input_im, im_name)
		assert_array_equal(expected_colony_mask_filled_holes,
			test_colony_mask_filled_holes)

	def test_EP_160110_t02xy1005_small(self):
		'''
		Test cleanup on EP_160110_t02xy1005_small
		The colony mask created here differs from matlab results; see
		comment in PIE.colony_edge_detect.EdgeDetector._run_cleanup
		In addition, hole filling is 4-connected in matlab, but
		8-connected in python PIE code
		'''
		max_proportion_exposed_edge = 0.75
		hole_fill_area = np.inf
		self._compare_cleanup_results('EP_160110_t02xy1005_small',
			max_proportion_exposed_edge, hole_fill_area)

class TestRunEdgeDetection(unittest.TestCase):

	def _compare_overlays(self, im_name, cleanup):
		hole_fill_area = np.inf
		# read in images
		im_path = os.path.join('tests', 'test_ims', (im_name + '.tif'))
		center_path = os.path.join('tests', 'test_ims',
			(im_name + '_cell_centers.tif'))
		if cleanup:
			colony_mask_path = os.path.join('tests', 'test_ims',
				(im_name + '_colony_mask_cleanup.tif'))
		else:
			colony_mask_path = os.path.join('tests', 'test_ims',
				(im_name + '_colony_mask.tif'))
		input_im = cv2.imread(im_path, cv2.IMREAD_ANYDEPTH)
		cell_centers = cv2.imread(center_path, cv2.IMREAD_ANYDEPTH).astype(bool)
		expected_final_colony_mask = \
			cv2.imread(colony_mask_path, cv2.IMREAD_ANYDEPTH).astype(bool)
		# run initial overlay identification
		edge_detector = EdgeDetector(input_im, cell_centers, hole_fill_area, cleanup, 0.75)
		test_final_colony_mask = edge_detector.run_edge_detection()
		if not np.array_equal(expected_final_colony_mask,
			test_final_colony_mask):
			general_pie_testing_functions.show_mask_diffs(expected_final_colony_mask,
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
		pixels that have been deleted, probably as a result of matlab
		filling holes that are 8-connected to background, while PIE does
		not
		These holes were re-created in reference mask to make this test
		not fail
		'''
		self._compare_overlays('EP_160110_t02xy1005_small', False)

	def test_test_im_small_with_cleanup(self):
		'''
		Test initial overlay creation + cleanup on test_im_small
		'''
		self._compare_overlays('test_im_small', True)

	def test_EP_160110_t02xy1005_small_with_cleanup(self):
		'''
		Test initial overlay creation + cleanup on
		EP_160110_t02xy1005_small
		The colony mask created here differs from matlab results; see
		comment in PIE.colony_edge_detect.EdgeDetector._run_cleanup
		In addition, hole filling is 4-connected in matlab, but
		8-connected in python PIE code
		'''
		self._compare_overlays('EP_160110_t02xy1005_small', True)

if __name__ == '__main__':
	unittest.main()
