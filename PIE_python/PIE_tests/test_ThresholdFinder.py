#!/usr/bin/python

import unittest
import cv2
import numpy as np
from PIE.colony_edge_detect import _ThresholdFinder
from numpy.testing import assert_array_equal, assert_allclose

class TestGetTophat(unittest.TestCase):

	def setUp(self):
		input_im = \
			cv2.imread('PIE_tests/test_ims/test_im_small.tif',
				cv2.IMREAD_ANYDEPTH)
		self.threshold_finder = _ThresholdFinder(input_im)
		self.expected_tophat = \
			cv2.imread('PIE_tests/test_ims/test_im_small_tophat.tif',
				cv2.IMREAD_ANYDEPTH)

	def test_get_tophat(self):
		'''
		Tests getting tophat of provided input image against matlab
		result
		'''
		self.threshold_finder._get_tophat()
		assert_array_equal(self.threshold_finder.tophat_im,
			self.expected_tophat)

class TestGetUniqueTophatVals(unittest.TestCase):

	def setUp(self):
		self.threshold_finder_standin = object.__new__(_ThresholdFinder)
		self.threshold_finder_standin.threshold_flag = 0

	def test_500_unique_vals(self):
		'''
		Tests that correct list of unique values returned for tophat
		array with 500 unique values, without raising threshold flag
		'''
		self.threshold_finder_standin.tophat_im = \
			np.array([range(1,301), range(201,501)])
		self.threshold_finder_standin._get_unique_tophat_vals()
		self.assertEqual(
			set(self.threshold_finder_standin.tophat_unique.flatten()),
			set(range(1,501)))
		self.assertEqual(self.threshold_finder_standin.threshold_flag, 0)

	def test_150_unique_vals(self):
		'''
		Tests that correct list of unique values returned for tophat
		array with 500 unique values
		'''
		self.threshold_finder_standin.tophat_im = \
			np.array([range(1,101), range(51,151)])
		self.threshold_finder_standin._get_unique_tophat_vals()
		self.assertEqual(
			set(self.threshold_finder_standin.tophat_unique.flatten()),
			set(range(1,151)))
		self.assertEqual(self.threshold_finder_standin.threshold_flag, 1)

	def test_3_unique_vals(self):
		'''
		Tests that error raised for tophat array with 3 unique values
		'''
		self.threshold_finder_standin.tophat_im = \
			np.array([[1, 2], [2, 3]])
		with self.assertRaises(ValueError):
			self.threshold_finder_standin._get_unique_tophat_vals()

class TestBinCentersToEdges(unittest.TestCase):

	def setUp(self):
		self.threshold_finder_standin = object.__new__(_ThresholdFinder)

	def test_bin_center_to_edge_conversion(self):
		'''
		Tests converting a list of unequally spaced bin centers to edges
		Conversion must reprodcue behavior of matlab hist function
		'''
		bin_centers = np.array([3, 4, 6.5])
		expected_bin_edges = np.array([-np.inf, 3.5, 5.25, np.inf])
		test_bin_edges = \
			self.threshold_finder_standin._bin_centers_to_edges(bin_centers)
		assert_array_equal(expected_bin_edges, test_bin_edges)

	def test_bin_center_to_edge_conversion_typeswap(self):
		'''
		Tests converting a list of unequally spaced bin centers to edges
		when bin centers are all ints but edges are floats
		'''
		bin_centers = np.array([2,7,12])
		expected_bin_edges = np.array([-np.inf, 4.5, 9.5, np.inf])
		test_bin_edges = \
			self.threshold_finder_standin._bin_centers_to_edges(bin_centers)
		assert_array_equal(expected_bin_edges, test_bin_edges)

class TestReproduceMatlabHist(unittest.TestCase):

	def setUp(self):
		self.threshold_finder_standin = object.__new__(_ThresholdFinder)
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
		(test_counts, test_centers) = \
			self.threshold_finder_standin._reproduce_matlab_hist(self.x, bins)
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
		test_counts, test_centers = \
			self.threshold_finder_standin._reproduce_matlab_hist(self.x, bins)
		assert_array_equal(expected_centers, test_centers)
		assert_array_equal(expected_counts, test_counts)

	def test_invalid_bins(self):
		'''
		Tests behavior of _reproduce_matlab_hist method with a list of
		bin centers (not allowed if list is not np array)
		'''
		bins = [.5, 1.5, 2.5]
		with self.assertRaises(TypeError):
			test_counts, test_centers = \
				self.threshold_finder_standin._reproduce_matlab_hist(self.x,
					bins)

class TestGetLogTophatHist(unittest.TestCase):

	def setUp(self):
		self.threshold_finder_standin = object.__new__(_ThresholdFinder)

	def test_synth_tophat_hist_bin_int(self):
		'''
		Tests _ThresholdFinder._get_log_tophat_hist on a made-up numpy
		array with an integer provided for bin number
		'''
		self.threshold_finder_standin.tophat_im = \
			np.array([[7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6]])
		tophat_bins = 4
		test_ln_tophat_hist, test_bin_centers = \
			self.threshold_finder_standin._get_log_tophat_hist(
				tophat_bins)
		expected_bin_centers = np.array([2.375, 5.125, 7.875, 10.625])
		expected_ln_hist = np.log([3, 3, 3, 3])
		assert_array_equal(expected_bin_centers, test_bin_centers)
		assert_array_equal(expected_ln_hist, test_ln_tophat_hist)

	def test_synth_tophat_hist_bin_list(self):
		'''
		Tests _ThresholdFinder._get_log_tophat_hist on a made-up numpy
		array with a list of bin centers provided
		'''
		self.threshold_finder_standin.tophat_im = \
			np.array([[7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6]])
		tophat_bins = np.array([2, 7, 12])
		expected_bin_centers = tophat_bins
		expected_ln_hist = np.log([4, 5, 3])
		test_ln_tophat_hist, test_bin_centers = \
			self.threshold_finder_standin._get_log_tophat_hist(
				tophat_bins)
		assert_array_equal(expected_bin_centers, test_bin_centers)
		assert_array_equal(expected_ln_hist, test_ln_tophat_hist)

	def test_synth_tophat_hist_bin_list_zero(self):
		'''
		Tests _ThresholdFinder._get_log_tophat_hist on a made-up numpy
		array with a list of bin centers provided, including a bin that
		contains 0 entities (expect a 0 at this position in the returned
		log histogram)
		'''
		self.threshold_finder_standin.tophat_im = \
			np.array([[7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6]])
		tophat_bins = np.array([2.9, 3.2, 3.5, 17.5])
		expected_bin_centers = tophat_bins
		expected_ln_hist = np.array([np.log(3), 0, np.log(7), np.log(2)])
		test_ln_tophat_hist, test_bin_centers = \
			self.threshold_finder_standin._get_log_tophat_hist(
				tophat_bins)
		assert_array_equal(expected_bin_centers, test_bin_centers)
		assert_allclose(expected_ln_hist, test_ln_tophat_hist, atol = 10**-10)

	def test_real_tophat_hist_bin_int(self):
		'''
		Compares output of _ThresholdFinder._get_log_tophat_hist to that
		of relevant PIE matlab code on a real image when an integer is
		provided for bin number
		'''
		self.threshold_finder_standin.tophat_im = \
			cv2.imread('PIE_tests/test_ims/test_im_small_tophat.tif',
				cv2.IMREAD_ANYDEPTH)
		tophat_bins = 10
		test_ln_tophat_hist, test_bin_centers = \
			self.threshold_finder_standin._get_log_tophat_hist(
				tophat_bins)
		expected_ln_hist = \
			np.array([10.470220808240414, 10.190957102406585, 6.003887067106539,
				3.433987204485146, 3.663561646129646, 3.295836866004329,
				3.258096538021482, 2.397895272798371, 3.367295829986474,
				3.044522437723423])
		expected_bin_centers = \
			np.array([56.8, 170.4, 284, 397.6, 511.2, 624.8, 738.4, 852, 965.6,
				1079.2])
		assert_allclose(expected_bin_centers, test_bin_centers,
			atol = 10**-10)
		assert_allclose(expected_ln_hist, test_ln_tophat_hist, atol = 10**-15)

	def test_real_tophat_hist_bin_list(self):
		'''
		Compares output of _ThresholdFinder._get_log_tophat_hist to that
		of relevant PIE matlab code on a real image when a list of bin
		centers is provided
		'''
		self.threshold_finder_standin.tophat_im = \
			cv2.imread('PIE_tests/test_ims/test_im_small_tophat.tif',
				cv2.IMREAD_ANYDEPTH)
		tophat_bins = np.array([250, 750, 1100])
		expected_bin_centers = tophat_bins
		expected_ln_hist = \
			np.array([11.040807602307288, 4.454347296253507, 3.828641396489095])
		test_ln_tophat_hist, test_bin_centers = \
			self.threshold_finder_standin._get_log_tophat_hist(
				tophat_bins)
		assert_allclose(expected_bin_centers, test_bin_centers,
			atol = 10**-10)
		assert_allclose(expected_ln_hist, test_ln_tophat_hist, atol = 10**-15)
		



if __name__ == '__main__':
	unittest.main()