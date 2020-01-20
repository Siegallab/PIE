#!/usr/bin/python

import unittest
import cv2
import numpy as np
from PIE.colony_edge_detect import _ThresholdFinder
from numpy.testing import assert_array_equal

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

if __name__ == '__main__':
	unittest.main()