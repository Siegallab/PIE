#!/usr/bin/python

import unittest
import numpy as np
from PIE import ported_matlab
from numpy.testing import assert_array_equal

class TestBinCentersToEdges(unittest.TestCase):

	def test_bin_center_to_edge_conversion(self):
		'''
		Tests converting a list of unequally spaced bin centers to edges
		Conversion must reprodcue behavior of matlab hist function
		'''
		bin_centers = np.array([3, 4, 6.5])
		expected_bin_edges = np.array([-np.inf, 3.5, 5.25, np.inf])
		test_bin_edges = ported_matlab._bin_centers_to_edges(bin_centers)
		assert_array_equal(expected_bin_edges, test_bin_edges)

	def test_bin_center_to_edge_conversion_typeswap(self):
		'''
		Tests converting a list of unequally spaced bin centers to edges
		when bin centers are all ints but edges are floats
		'''
		bin_centers = np.array([2,7,12])
		expected_bin_edges = np.array([-np.inf, 4.5, 9.5, np.inf])
		test_bin_edges = ported_matlab._bin_centers_to_edges(bin_centers)
		assert_array_equal(expected_bin_edges, test_bin_edges)

class TestHist(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		# Unable to reproduce matlab behavior for an array x with int
		# values on the edges of the bins because matlab behaves
		# unpredicatably in those cases
		#self.x = np.array([0, 0, 2, 3, 0, 2])
		self.x = np.array([0, 0.3, 2.1, 3, 0, 2.5])

	def test_int_bins(self):
		'''
		Tests behavior of _reproduce_matlab_hist method with an integer
		bin number
		'''
		bins = 3
		expected_centers = np.array([.5, 1.5, 2.5])
		expected_counts = np.array([3, 0, 3])
		(test_counts, test_centers) = ported_matlab.hist(self.x, bins)
		assert_array_equal(expected_centers, test_centers)
		assert_array_equal(expected_counts, test_counts)

	def test_bin_center_array(self):
		'''
		Tests behavior of _reproduce_matlab_hist method with an np array
		of bin centers
		'''
		bins = np.array([.5, 1.5, 2.5])
		expected_centers = bins
		expected_counts = np.array([3, 0, 3])
		test_counts, test_centers = ported_matlab.hist(self.x, bins)
		assert_array_equal(expected_centers, test_centers)
		assert_array_equal(expected_counts, test_counts)

	def test_invalid_bins(self):
		'''
		Tests behavior of _reproduce_matlab_hist method with a list of
		bin centers (not allowed if list is not np array)
		'''
		bins = [.5, 1.5, 2.5]
		with self.assertRaises(TypeError):
			test_counts, test_centers = ported_matlab.hist(self.x,
					bins)

class TestBwperim(unittest.TestCase):

	def test_bwperim_full(self):
		'''
		Checks that perim_mask returned for an input mask is the same
		as that returned by matlab with default connectivity
		'''
		input_mask = np.array([
			[False, False, False,  True,  True,  True],
			[False, False, False,  True,  True,  True],
			[ True,  True,  True,  True,  True,  True],
			[ True,  True,  True,  True,  True, False],
			[ True,  True,  True,  True, False, False],
			[ True,  True,  True, False, False, False]])
		expected_perim_mask = np.array([
			[False, False, False, True, True, True],
			[False, False, False, True, False, True],
			[True, True, True, False, False, True],
			[True, False, False, False, True, False],
			[True, False, False, True, False, False],
			[True, True, True, False, False, False]])
		test_perim_mask = ported_matlab.bwperim(input_mask)
		assert_array_equal(expected_perim_mask, test_perim_mask)

	def test_bwperim_holes(self):
		'''
		Checks that perim_mask returned for an input mask with holes in
		it is the same as that returned by matlab with default
		connectivity
		'''
		input_mask = np.array([
			[False, False, False,  True,  True,  True],
			[False, False, False,  True,  True,  True],
			[ True,  True,  True,  True,  True,  True],
			[ True,  False,  True,  True,  True, False],
			[ True,  False,  True,  True, False, False],
			[ True,  True,  True, False, False, False]])
		expected_perim_mask = np.array([
			[False, False, False, True, True, True],
			[False, False, False, True, False, True],
			[True, True, True, False, False, True],
			[True, False, True, False, True, False],
			[True, False, True, True, False, False],
			[True, True, True, False, False, False]])
		test_perim_mask = ported_matlab.bwperim(input_mask)
		assert_array_equal(expected_perim_mask, test_perim_mask)

class TestBwareaopen(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.input_mask = np.array([
			[False, False, False,  True,  True,  True],
			[False, False, False,  True,  False,  True],
			[ True,  True,  False,  True,  True,  True],
			[ True,  True,  True,  False,  False, False],
			[ True,  False,  True,  True, False, False],
			[ True,  True,  True, False, False, False]])

	def test_bwareaopen_conn_4(self):
		'''
		Checks that when connecticity is set to 4, object
		whose area is same as P is removed from final mask
		'''
		P = 8
		expected_filled_mask = np.array([
			[False, False, False,  False,  False,  False],
			[False, False, False,  False,  False,  False],
			[ True,  True,  False,  False,  False,  False],
			[ True,  True,  True,  False,  False, False],
			[ True,  False,  True,  True, False, False],
			[ True,  True,  True, False, False, False]])
		test_filled_mask = ported_matlab.bwareaopen(self.input_mask, P, conn = 4)
		assert_array_equal(expected_filled_mask, test_filled_mask)

	def test_bwareaopen_conn_8(self):
		'''
		Checks that when connecticity is set to 8 (default), object
		whose area is same as P is removed from final mask
		'''
		P = 8
		expected_filled_mask = self.input_mask
		test_filled_mask = \
			ported_matlab.bwareaopen(self.input_mask, P)
		assert_array_equal(expected_filled_mask, test_filled_mask)

if __name__ == '__main__':
	unittest.main()