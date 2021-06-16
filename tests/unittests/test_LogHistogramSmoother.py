#!/usr/bin/python

import unittest
import numpy as np
from PIE.adaptive_threshold import _LogHistogramSmoother
from numpy.testing import assert_array_equal, assert_allclose

class TestSetSmoothingWindowSize(unittest.TestCase):

	def setUp(self):
		self.log_hist_smoother_standin = object.__new__(_LogHistogramSmoother)
		self.default_window_size = 21

	def test_default_window_size(self):
		'''
		Tests default window size is selected when ln_tophat_hist is
		long enough
		'''
		ln_tophat_hist = range(0, 64)
		expected_window_size = 21
		test_smooth_window_size = \
			self.log_hist_smoother_standin._set_smoothing_window_size(
				ln_tophat_hist, self.default_window_size)
		self.assertEqual(expected_window_size, test_smooth_window_size)

	def test_window_size_reductions(self):
		'''
		Test correct rounding to nearest odd number when default window
		size is too large
		'''
		hist_lengths_window_sizes = \
			[(21, 7), (20, 7), (19, 7), (18, 7), (17, 5), (16, 5), (15, 5)]
		for hist_length, expected_window_size in hist_lengths_window_sizes:
			ln_tophat_hist = range(0, hist_length)
			test_smooth_window_size = \
				self.log_hist_smoother_standin._set_smoothing_window_size(
					ln_tophat_hist, self.default_window_size)
			self.assertEqual(expected_window_size, test_smooth_window_size)

class TestSmoothLogHistogram(unittest.TestCase):

	def setUp(self):
		self.log_hist_smoother_standin = object.__new__(_LogHistogramSmoother)
		hist_array_data = \
			np.loadtxt('tests/test_ims/tophat_im_small_best_hist.csv',
				delimiter=',')
		self.ln_tophat_hist = hist_array_data[1]
		self.expected_ln_tophat_smooth = hist_array_data[2]
		self.window_size = 21

	def test_histogram_smoothing(self):
		'''
		Test histogram smoothing on array (which needs code to perform
		window size adjustment)
		'''
		test_ln_tophat_smooth = \
			self.log_hist_smoother_standin._smooth_log_histogram(
				self.ln_tophat_hist, self.window_size)
		assert_allclose(self.expected_ln_tophat_smooth,
			test_ln_tophat_smooth,
			atol = 10**-10)

if __name__ == '__main__':
	unittest.main()