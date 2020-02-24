#!/usr/bin/python

'''
Tests that thresholds for input histograms produce correct results
'''

import unittest
import os
import numpy as np
from PIE import adaptive_threshold
from numpy.testing import assert_allclose

#from plotnine import *
#import pandas as pd

# Old SFig2A: 'xy01_08ms_3702': 11752
# Old SFig2B: 't10xy0320': 2662.2
# Old SFig2C: 'xy01_14ms_3702': 14240
# Old SFig2D: 't02xy0225': 10338
# Old SFig2E: 't09xy1107': 6816

class Test_mu1PosThresholdMethod(unittest.TestCase):
	'''
	Tests images that were run through _mu1PosThresholdMethod-equivalent
	method in matlab, and compares thresholds arrived at there with the
	ones identified by PIE.adaptive_threshold
	'''

	@classmethod
	def setUpClass(self):
		# set relative fold tolerance for comparison of expected and
		# actual thresholds
		self.rel_tolerance = 0.2

	def _get_threshold(self, im_name):
		'''
		Opens histogram file (calculated in matlab) corresponding to
		im_name, reads in smoothed log histogram, calculates and returns
		threshold via PIE.adaptive_threshold._mu1PosThresholdMethod
		'''
		im_path = os.path.join('PIE_tests', 'test_ims',
			(im_name + '_best_hist.csv'))
		hist_data = np.loadtxt(im_path, delimiter=',')
		x_pos = hist_data[0]
		ln_hist_smooth = hist_data[2]
		threshold_method = \
			adaptive_threshold._mu1PosThresholdMethod(x_pos, ln_hist_smooth)
		threshold = threshold_method.get_threshold()
		return(threshold)

	def test_xy01_08ms_3702(self):
		test_threshold = self._get_threshold('xy01_08ms_3702')
		expected_threshold = 11752
		assert_allclose(expected_threshold, test_threshold,
			rtol = self.rel_tolerance)

	def test_t10xy0320(self):
		test_threshold = self._get_threshold('t10xy0320')
		expected_threshold = 2662.2
		assert_allclose(expected_threshold, test_threshold,
			rtol = self.rel_tolerance)

	def test_xy01_14ms_3702(self):
		test_threshold = self._get_threshold('xy01_14ms_3702')
		expected_threshold = 14240
		assert_allclose(expected_threshold, test_threshold,
			rtol = self.rel_tolerance)

class Test_mu1ReleasedThresholdMethod(unittest.TestCase):
	'''
	Tests images that were run through _mu1ReleasedThresholdMethod-
	equivalent method in matlab, and compares thresholds arrived at
	there with the ones identified by PIE.adaptive_threshold
	'''

	@classmethod
	def setUpClass(self):
		# set relative fold tolerance for comparison of expected and
		# actual thresholds
		self.rel_tolerance = 0.2

	def _get_threshold(self, im_name):
		'''
		Opens histogram file (calculated in matlab) corresponding to
		im_name, reads in smoothed log histogram, calculates and returns
		threshold via PIE.adaptive_threshold._mu1PosThresholdMethod
		'''
		im_path = os.path.join('PIE_tests', 'test_ims',
			(im_name + '_best_hist.csv'))
		hist_data = np.loadtxt(im_path, delimiter=',')
		x_pos = hist_data[0]
		ln_hist_smooth = hist_data[2]
		threshold_method = \
			adaptive_threshold._mu1ReleasedThresholdMethod(x_pos,
				ln_hist_smooth)
		threshold = threshold_method.get_threshold()
		return(threshold)

	def test_t02xy0225(self):
		test_threshold = self._get_threshold('t02xy0225')
		expected_threshold = 10338
		assert_allclose(expected_threshold, test_threshold,
			rtol = self.rel_tolerance)

	def test_t10xy0320(self):
		'''
		This histogram would be sent to sliding circle method
		'''
		test_threshold = self._get_threshold('t10xy0320')
		expected_threshold = np.nan
		assert_allclose(expected_threshold, test_threshold,
			rtol = self.rel_tolerance)

class Test_DataSlidingCircleThresholdMethod(unittest.TestCase):
	'''
	Tests that threshold calculated for images by slide_circle_data
	function in matlab PIE code is close to the one calculated by
	PIE.adaptive_threshold._DataSlidingCircleThresholdMethod
	'''

	@classmethod
	def setUpClass(self):
		# set relative fold tolerance for comparison of expected and
		# actual thresholds
		self.rel_tolerance = 0.3

	def _get_threshold(self, im_name):
		'''
		Opens histogram file (calculated in matlab) corresponding to
		im_name, reads in smoothed log histogram, calculates and returns
		threshold via PIE.adaptive_threshold._mu1PosThresholdMethod
		'''
		im_path = os.path.join('PIE_tests', 'test_ims',
			(im_name + '_best_hist.csv'))
		hist_data = np.loadtxt(im_path, delimiter=',')
		x_pos = hist_data[0]
		ln_hist = hist_data[1]
		threshold_method = \
			adaptive_threshold._DataSlidingCircleThresholdMethod(x_pos, ln_hist)
		threshold = threshold_method.get_threshold()
#		print('\n\n')
#		print(im_name)
#		print(threshold)
#		output_df = pd.DataFrame(
#			{'x_vals': (threshold_method._x_vals_stretched / threshold_method._x_stretch_factor),
#			'y_vals': (threshold_method._y_vals_stretched / threshold_method._y_stretch_factor)})
#		p = ggplot(output_df) + geom_line(aes(x = 'x_vals', y = 'y_vals')) + geom_vline(aes(xintercept = threshold))
#		ggsave(plot = p, filename = os.path.join('PIE_tests', 'test_ims',
#			(im_name + '_plot.pdf')))
		return(threshold)

	def test_xy01_08ms_3702(self):
		# 12608
		test_threshold = self._get_threshold('xy01_08ms_3702')
		expected_threshold = 9312
		assert_allclose(expected_threshold, test_threshold,
			rtol = self.rel_tolerance)

#	def test_t10xy0320(self):
#		'''
#		Here, threshold differs a lot from that calculated via gaussians
#		because default lower bound is higher than optimal threshold
#		'''
#		# 25533.0
#		test_threshold = self._get_threshold('t10xy0320')
#		expected_threshold = 7686
#		assert_allclose(expected_threshold, test_threshold,
#			rtol = self.rel_tolerance)

	def test_xy01_14ms_3702(self):
		# 13360
		test_threshold = self._get_threshold('xy01_14ms_3702')
		expected_threshold = 13360
		assert_allclose(expected_threshold, test_threshold,
			rtol = self.rel_tolerance)

	def test_t02xy0225(self):
		# 9422.8
		test_threshold = self._get_threshold('t02xy0225')
		expected_threshold = 9024
		assert_allclose(expected_threshold, test_threshold,
			rtol = self.rel_tolerance)

	def test_t09xy1107(self):
		# 7141.8
		test_threshold = self._get_threshold('t09xy1107')
		expected_threshold = 7142
		assert_allclose(expected_threshold, test_threshold,
			rtol = self.rel_tolerance)