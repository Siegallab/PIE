#!/usr/bin/python

import unittest
import numpy as np
import sys
import warnings
from scipy.optimize import least_squares
from PIE.colony_edge_detect import _GaussianFitThresholdMethod
from numpy.testing import assert_array_equal, assert_allclose

class TestCheckThresholds(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.gausian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)
		self.gausian_threshold_standin.param_idx_dict = \
			{'alpha': 0, 'beta': 1, 'gamma': 2, 'delta': 3, 'epsilon': 4}
		self.gausian_threshold_standin.non_neg_params = ['gamma', 'epsilon']
		self.gausian_threshold_standin.above_zero_params =  ['alpha']

	def test_correct_bounds(self):
		'''
		Tests that when bounds are provided without any values below 0
		at self.non_negative_params, unmodified np array returned
		'''
		input_bounds = np.array([2, 0, 3.5, -1.3, 4])
		test_bounds = self.gausian_threshold_standin._check_bounds(input_bounds)
		assert_array_equal(input_bounds, test_bounds)

	def test_neg_bounds(self):
		'''
		Tests that when bounds are provided withvalues below 0 at
		self.non_negative_params, and 0 at self.above_zero_params,
		they are replaced by 0
		'''
		input_bounds = np.array([-1.2, 0, 3.5, -1.3, -1])
		expected_bounds = np.array([sys.float_info.min, 0, 3.5, -1.3, 0])
		with warnings.catch_warnings(record=True) as w:
			# Cause all warnings to always be triggered.
			warnings.simplefilter("always")
			test_bounds = \
				self.gausian_threshold_standin._check_bounds(input_bounds)
			# Check that 2 warnings issued
			assert len(w) == 2
			assert issubclass(w[-1].category, UserWarning)
		assert_array_equal(expected_bounds, test_bounds)

	def test_list_bounds(self):
		'''
		Tests that when bounds are provided as a list (rather than np
		array), error is raised
		'''
		input_bounds = [2, 0, 3.5, -1.3, 4]
		with self.assertRaises(TypeError):
			test_bounds = self.gausian_threshold_standin._check_bounds(input_bounds)
		
	def test_float_bounds(self):
		'''
		Tests that when bounds are provided as a float, error is raised
		'''
		input_bounds = 3.3
		with self.assertRaises(TypeError):
			test_bounds = self.gausian_threshold_standin._check_bounds(input_bounds)

	def test_wrong_length_bounds(self):
		'''
		Tests that when bounds of the wrong length are provided, error
		is raised
		'''
		input_bounds = np.array([0, 3.5, -1.3, 4])
		with self.assertRaises(TypeError):
			test_bounds = self.gausian_threshold_standin._check_bounds(input_bounds)

class TestIDStartingVals(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.array_data = \
			np.loadtxt('PIE_tests/test_ims/tophat_im_small_best_hist.csv',
				delimiter=',')

	def test_small_im_hist(self):
		'''
		Tests identification of starting parameters on smoothed
		histogram of small test image
		'''
		self.gaussian_method = \
			_GaussianFitThresholdMethod('test', self.array_data[0],
				self.array_data[2], np.ones(6), np.ones(6))
		expected_mu1 = 54.095238095238
		expected_sigma1 = 90.1587301587302
		expected_lambda1 = 10.4417048619897
		expected_mu2 = 540.952380952381
		expected_sigma2 = 180.31746031746
		expected_lambda2 = 2.63649740322885
		expected_starting_param_vals = \
			np.array([expected_lambda1, expected_mu1, expected_sigma1,
				expected_lambda2, expected_mu2, expected_sigma2])
		self.gaussian_method._id_starting_vals()
		assert_allclose(expected_starting_param_vals,
			self.gaussian_method.starting_param_vals)

class TestDigaussCalculator(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.gausian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)
		self.test_x = np.array([-3, -2, -1, 0, 1, 2, 3])

	def test_digauss_calc(self):
		'''
		Tests calculation of gaussian mixture with reasonable input
		parameters
		'''
		lambda_1 = 1
		mu_1 = 0
		sigma_1 = 1
		lambda_2 = .5
		mu_2 = -1
		sigma_2 = 2
		# expected_y from matlab code
		expected_y = \
			np.array(
				[0.184063130389808, 0.407716030424437, 0.867879441171442,
				1.3894003915357, 0.551819161757164, 0.0710152511696663,
				0.00928122924845377])
		test_y = self.gausian_threshold_standin._digauss_calculator(self.test_x,
			lambda_1, mu_1, sigma_1, lambda_2, mu_2, sigma_2)
		assert_allclose(expected_y, test_y)

	def test_digauss_calc_extreme_vals(self):
		'''
		Tests calculation of gaussian mixture with extreme input
		parameters
		'''
		lambda_1 = 1
		mu_1 = 0
		sigma_1 = sys.float_info.min
		lambda_2 = 0
		mu_2 = -1
		sigma_2 = 2
		# expected_y from matlab code
		expected_y = np.array([0, 0, 0, 1, 0, 0, 0])
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			test_y = self.gausian_threshold_standin._digauss_calculator(
				self.test_x, lambda_1, mu_1, sigma_1, lambda_2, mu_2, sigma_2)
		assert_allclose(expected_y, test_y)

class TestDigaussianResidualFunCalculator(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.gausian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)
		self.gausian_threshold_standin.param_idx_dict = \
			{'lambda_1': 0, 'mu_1': 1, 'sigma_1': 2, 'lambda_2': 3, 'mu_2': 4,
				'sigma_2': 5}
		self.test_x = np.array([-3, -2, -1, 0, 1, 2, 3])
		self.test_y = np.array([1, 1, 1, 1, 1, 1, 1])

	def test_residual_calc(self):
		'''
		Tests calculation of residuals between gaussian mixture with
		reasonable input parameters and vector of ones
		'''
		lambda_1 = 1
		mu_1 = 0
		sigma_1 = 1
		lambda_2 = .5
		mu_2 = -1
		sigma_2 = 2
		params = np.array([lambda_1, mu_1, sigma_1, lambda_2, mu_2, sigma_2])
		# resid from y expected by matlab code
		expected_resid = \
			np.array([0.81593687, 0.59228397, 0.13212056, -0.38940039,
				0.44818084, 0.92898475, 0.99071877])
		test_resid = self.gausian_threshold_standin._digaussian_residual_fun(
			params, self.test_x, self.test_y)
		assert_allclose(expected_resid, test_resid)

	def test_residual_calc_extreme_vals(self):
		'''
		Tests calculation of calculation of residuals between gaussian
		mixture with extreme input parameters and vector of ones
		'''
		lambda_1 = 1
		mu_1 = 0
		sigma_1 = sys.float_info.min
		lambda_2 = 0
		mu_2 = -1
		sigma_2 = 2
		params = np.array([lambda_1, mu_1, sigma_1, lambda_2, mu_2, sigma_2])
		# resid from y expected by matlab code
		expected_resid = np.array([1, 1, 1, 0, 1, 1, 1])
		test_resid = self.gausian_threshold_standin._digaussian_residual_fun(
			params, self.test_x, self.test_y)
		assert_allclose(expected_resid, test_resid)

class TestFitGaussians(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		array_data = \
			np.loadtxt('PIE_tests/test_ims/tophat_im_small_best_hist.csv',
				delimiter=',')
		self.x = array_data[0]
		self.y = array_data[2]

	def setUp(self):
		self.gausian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)
		self.gausian_threshold_standin.param_idx_dict = \
			{'lambda_1': 0, 'mu_1': 1, 'sigma_1': 2, 'lambda_2': 3, 'mu_2': 4,
				'sigma_2': 5}

	def test_digaussian_fit(self):
		'''
		Tests that fit to test data closely matches that found by matlab
		(which finds a good fit to this data) when done with same
		starting parameters and bounds
		'''
		starting_param_vals = \
			np.array([10.4417048619897, 81.1428571428571, 113.90566370064,
				4.96075048462661, 243.428571428571, 157.449461988403])
		self.gausian_threshold_standin.lower_bounds = \
			np.array([1, 0, 0, 0.5, -np.inf, 0])
		self.gausian_threshold_standin.upper_bounds = np.array([np.inf]*6)
		expected_sse = 0.71
		self.gausian_threshold_standin._fit_gaussians(starting_param_vals,
			self.x, self.y)
		test_sse = self.gausian_threshold_standin.fit_results.cost
		# check that the cost in the python fit is less than 1.25x of
		# the cost of the matlab fit
		self.assertTrue(test_sse < expected_sse * 1.25)

class TestCalcFitAdjRsq(unittest.TestCase):

	def setUp(self):
		self.gausian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)

	def _regression_model(self, params, x, y):
		'''
		Linear model to create least squares fit
		'''
		yhat = params[0]*x + params[1]
		res = y - yhat
		return(res)

	def test_adjusted_rsq(self):
		'''
		Run linear model on some fake data, calculate adj r squared
		'''
		self.gausian_threshold_standin.x = np.arange(0,10)
		self.gausian_threshold_standin.y = np.arange(0,10)*2+3
		self.gausian_threshold_standin.y[[3,5]] = [10,12]
		self.gausian_threshold_standin.fit_results = \
			least_squares(self._regression_model, [2,3],
				args=(self.gausian_threshold_standin.x,
					self.gausian_threshold_standin.y))
		expected_adj_rsq = 0.9922558922558923
		self.gausian_threshold_standin._calc_fit_adj_rsq()
		self.assertEqual(expected_adj_rsq,
			self.gausian_threshold_standin.rsq_adj)

class TestFindPeakXPos(unittest.TestCase):

	def setUp(self):
		self.gausian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)

	def test_single_peak_finder(self):
		'''
		Tests finding x-position corresponding to single peak of y_hat
		'''
		x = np.array([0, 0.3, 1, 2, 3.5])
		y = np.array([1]*len(x))
		y_hat = np.array([0, 1.5, 1.3, .9, 1.4])
		residuals = y - y_hat
		self.gausian_threshold_standin._find_peak_x_pos(x, y, residuals)
		self.assertEqual(0.3, self.gausian_threshold_standin.peak_x_pos)

	def test_double_peak_finder(self):
		'''
		Tests finding x-position corresponding to first peak of y_hat
		'''
		x = np.array([0, 0.3, 1, 2, 3.5])
		y = np.array([1]*len(x))
		y_hat = np.array([0, 1.5, 1.3, .9, 1.5])
		residuals = y - y_hat
		self.gausian_threshold_standin._find_peak_x_pos(x, y, residuals)
		self.assertEqual(0.3, self.gausian_threshold_standin.peak_x_pos)


if __name__ == '__main__':
	unittest.main()