#!/usr/bin/python

import unittest
import cv2
import numpy as np
from numpy.testing import assert_array_equal
from PIE.colony_edge_detect import _PiePiece

class TestInit(unittest.TestCase):

	def test_init(self):
		'''
		Test initialization of _PiePiece object
		'''
		Gx = np.genfromtxt('PIE_tests/test_ims/test_im_small_Gx.csv',
			delimiter=',')
		Gy = np.genfromtxt('PIE_tests/test_ims/test_im_small_Gy.csv',
			delimiter=',')
		expected_pie_i_overlay = \
			cv2.imread('PIE_tests/test_ims/test_im_small_pie_pieces_i.tif',
			cv2.IMREAD_ANYDEPTH).astype(bool)
		Gx_right = Gx < 0
		Gy_top = Gy > 0
		pie_piece_i = _PiePiece(Gx_right, Gy_top)
		assert_array_equal(expected_pie_i_overlay,
            pie_piece_i.pie_mask)

class TestIDCenterOverlappingPieces(unittest.TestCase):

    def setUp(self):
        '''
        Create a list of tuples containing the input pie piece mask and
        the output of those intersecting the cell centers for each of
        the 4 PIE quadrants
        '''
        self.pie_piece_standin = object.__new__(_PiePiece)
        self.cell_centers = \
            cv2.imread('PIE_tests/test_ims/test_im_small_cell_centers.tif',
                cv2.IMREAD_ANYDEPTH).astype(bool)
        pie_quadrants = ['i', 'ii', 'iii', 'iv']
        self.input_and_output_pie_quadrants = []
        for current_quadrant in pie_quadrants:
            pie_quadrant = \
                cv2.imread('PIE_tests/test_ims/test_im_small_pie_pieces_' +
                    current_quadrant + '.tif', cv2.IMREAD_ANYDEPTH).astype(bool)
            pie_quadrant_selected = \
                cv2.imread(
                    'PIE_tests/test_ims/test_im_small_pie_pieces_center_overlap_' +
                    current_quadrant + '.tif', cv2.IMREAD_ANYDEPTH).astype(bool)
            current_input_output_tuple = \
                tuple([pie_quadrant, pie_quadrant_selected])
            self.input_and_output_pie_quadrants.append(
                current_input_output_tuple)

    def test_ID_overlaps(self):
        '''
        Tests that cell center-overlapping PIE pieces correctly
        identified in all quadrants
        '''
        for pie_mask, cell_intersect_pie_mask in \
            self.input_and_output_pie_quadrants:
            self.pie_piece_standin.pie_mask = pie_mask
            test_cell_intersect_pie_mask = \
                self.pie_piece_standin.id_center_overlapping_pie_pieces(
                    self.cell_centers)
            assert_array_equal(cell_intersect_pie_mask,
                test_cell_intersect_pie_mask)


if __name__ == '__main__':
	unittest.main()