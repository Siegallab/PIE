#!/usr/bin/python

import unittest
import numpy as np
import sys
import warnings
from scipy.optimize import least_squares
from PIE.adaptive_threshold import _GaussianFitThresholdMethod, \
	_mu1PosThresholdMethod, _mu1ReleasedThresholdMethod, \
	_SlidingCircleThresholdMethod
from numpy.testing import assert_array_equal, assert_allclose

def _regression_model(params, x, y):
	'''
	Linear model to create least squares fit
	'''
	y_hat = params[0]*x + params[1]
	res = y - y_hat
	return(res)

### unittests for _GaussianFitThresholdMethod ###

class TestCheckThresholds(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.gaussian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)
		self.gaussian_threshold_standin.param_idx_dict = \
			{'alpha': 0, 'beta': 1, 'gamma': 2, 'delta': 3, 'epsilon': 4}
		self.gaussian_threshold_standin.non_neg_params = ['gamma', 'epsilon']
		self.gaussian_threshold_standin.above_zero_params =  ['alpha']

	def test_correct_bounds(self):
		'''
		Tests that when bounds are provided without any values below 0
		at self.non_negative_params, unmodified np array returned
		'''
		input_bounds = np.array([2, 0, 3.5, -1.3, 4])
		test_bounds = self.gaussian_threshold_standin._check_bounds(input_bounds)
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
				self.gaussian_threshold_standin._check_bounds(input_bounds)
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
			test_bounds = self.gaussian_threshold_standin._check_bounds(input_bounds)
		
	def test_float_bounds(self):
		'''
		Tests that when bounds are provided as a float, error is raised
		'''
		input_bounds = 3.3
		with self.assertRaises(TypeError):
			test_bounds = self.gaussian_threshold_standin._check_bounds(input_bounds)

	def test_wrong_length_bounds(self):
		'''
		Tests that when bounds of the wrong length are provided, error
		is raised
		'''
		input_bounds = np.array([0, 3.5, -1.3, 4])
		with self.assertRaises(TypeError):
			test_bounds = self.gaussian_threshold_standin._check_bounds(input_bounds)

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
			_GaussianFitThresholdMethod('test', 0, self.array_data[0],
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
		self.gaussian_threshold_standin = \
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
		test_y = self.gaussian_threshold_standin._digauss_calculator(self.test_x,
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
			test_y = self.gaussian_threshold_standin._digauss_calculator(
				self.test_x, lambda_1, mu_1, sigma_1, lambda_2, mu_2, sigma_2)
		assert_allclose(expected_y, test_y)

class TestDigaussianResidualFunCalculator(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.gaussian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)
		self.gaussian_threshold_standin.param_idx_dict = \
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
		test_resid = self.gaussian_threshold_standin._digaussian_residual_fun(
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
		test_resid = self.gaussian_threshold_standin._digaussian_residual_fun(
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
		self.gaussian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)
		self.gaussian_threshold_standin.param_idx_dict = \
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
		self.gaussian_threshold_standin.lower_bounds = \
			np.array([1, 0, 0, 0.5, -np.inf, 0])
		self.gaussian_threshold_standin.upper_bounds = np.array([np.inf]*6)
		expected_sse = 0.71
		self.gaussian_threshold_standin._fit_gaussians(starting_param_vals,
			self.x, self.y)
		test_sse = self.gaussian_threshold_standin.fit_results.cost
		# check that the cost in the python fit is less than 1.25x of
		# the cost of the matlab fit
		self.assertTrue(test_sse < expected_sse * 1.25)
		# check that self.gaussian_threshold_standin.y_hat is the output
		# of applying the model to the x values
		expected_y_hat = self.gaussian_threshold_standin._digauss_calculator(
			self.x, *self.gaussian_threshold_standin.fit_results.x)
		assert_allclose(expected_y_hat, self.gaussian_threshold_standin.y_hat)

class TestCalcFitAdjRsq(unittest.TestCase):

	def setUp(self):
		self.gaussian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)

	def test_adjusted_rsq(self):
		'''
		Run linear model on some fake data, calculate adj r squared
		'''
		self.gaussian_threshold_standin.x = np.arange(0,10)
		self.gaussian_threshold_standin.y = np.arange(0,10)*2+3
		self.gaussian_threshold_standin.y[[3,5]] = [10,12]
		self.gaussian_threshold_standin.fit_results = \
			least_squares(_regression_model, [2,3],
				args=(self.gaussian_threshold_standin.x,
					self.gaussian_threshold_standin.y))
		expected_adj_rsq = 0.9922558922558923
		self.gaussian_threshold_standin._calc_fit_adj_rsq()
		self.assertEqual(expected_adj_rsq,
			self.gaussian_threshold_standin.rsq_adj)

class TestFindPeak(unittest.TestCase):

	def setUp(self):
		self.gaussian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)

	def test_single_peak_finder(self):
		'''
		Tests finding x-position corresponding to single peak of y_hat
		'''
		self.gaussian_threshold_standin.x = np.array([0, 0.3, 1, 2, 3.5])
		self.gaussian_threshold_standin.y = \
			np.array([1]*len(self.gaussian_threshold_standin.x))
		self.gaussian_threshold_standin.y_hat = np.array([0, 1.5, 1.3, .9, 1.4])
		self.gaussian_threshold_standin._find_peak()
		self.assertEqual(0.3, self.gaussian_threshold_standin.peak_x_pos)
		self.assertEqual(1.5, self.gaussian_threshold_standin.y_peak_height)

	def test_double_peak_finder(self):
		'''
		Tests finding x-position corresponding to first peak of y_hat
		'''
		self.gaussian_threshold_standin.x = np.array([0, 0.3, 1, 2, 3.5])
		self.gaussian_threshold_standin.y = \
			np.array([1]*len(self.gaussian_threshold_standin.x))
		self.gaussian_threshold_standin.y_hat = np.array([0, 1.5, 1.3, .9, 1.5])
		self.gaussian_threshold_standin._find_peak()
		self.assertEqual(0.3, self.gaussian_threshold_standin.peak_x_pos)
		self.assertEqual(1.5, self.gaussian_threshold_standin.y_peak_height)

class TestGenerateFitResultDict(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.gaussian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)
		self.gaussian_threshold_standin.param_idx_dict = \
			{'mu': 0, 'sigma': 1}
		self.gaussian_threshold_standin.x = np.arange(0,10)
		self.gaussian_threshold_standin.y = np.arange(0,10)*2+3
		self.gaussian_threshold_standin.y[[3,5]] = [10,12]
		self.gaussian_threshold_standin.fit_results = \
			least_squares(_regression_model, [2,3],
				args=(self.gaussian_threshold_standin.x,
					self.gaussian_threshold_standin.y))

	def test_generate_fit_result_dict(self):
		'''
		Test generation of dictionary with parameter name keys from fit
		results
		'''
		self.gaussian_threshold_standin._generate_fit_result_dict()
		self.assertEqual(self.gaussian_threshold_standin.fit_results.x[0],
			self.gaussian_threshold_standin.fit_result_dict['mu'])
		self.assertEqual(self.gaussian_threshold_standin.fit_results.x[1],
			self.gaussian_threshold_standin.fit_result_dict['sigma'])

class TestCalcTypicalThreshold(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.gaussian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)
		self.gaussian_threshold_standin.fit_result_dict = \
			{'mu_1': 0, 'sigma_1': 2, 'mu_2': -1, 'sigma_2': 0.7}

	def test_calc_threshold_1(self):
		'''
		Tests calculation of threshold based on mu_1+2*sigma_1
		'''
		expected_threshold = 4
		test_threshold = \
			self.gaussian_threshold_standin._calc_typical_threshold(1)
		assert_allclose(expected_threshold, test_threshold)

	def test_calc_threshold_2(self):
		'''
		Tests calculation of threshold based on mu_2+2*sigma_2
		'''
		expected_threshold = .4
		test_threshold = \
			self.gaussian_threshold_standin._calc_typical_threshold(2)
		assert_allclose(expected_threshold, test_threshold)

class TestCalcMuDistanceToPeak(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.gaussian_threshold_standin = \
			object.__new__(_GaussianFitThresholdMethod)
		self.gaussian_threshold_standin.fit_result_dict = \
			{'mu_1': 0, 'mu_2': 2}
		self.gaussian_threshold_standin.peak_x_pos = 0.3

	def test_mu_distance_to_peak(self):
		'''
		Tests calculation of absolute value of distance of mu values to
		peak x positions
		'''
		expected_distances = np.array([0.3, 1.7])
		test_distances = \
			self.gaussian_threshold_standin._calc_mu_distance_to_peak()
		assert_allclose(expected_distances, test_distances)

class TestFitThresholdWithDistantPeaks(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.array_data = \
			np.loadtxt('PIE_tests/test_ims/tophat_im_small_best_hist.csv',
				delimiter=',')

	def setUp(self):
		self.gaussian_method = \
			_GaussianFitThresholdMethod('test', 0, self.array_data[0],
				self.array_data[2], np.ones(6), np.ones(6))
		# self.gaussian_method._min_real_peak_x_pos = 2.772380952380952
		self.gaussian_method._close_to_peak_dist = 2000

	def test_threshold_based_on_mu_1(self):
		'''
		Calculate threshold based on mu, with the first distribution
		used for threshold calculation
		'''
		self.gaussian_method.peak_x_pos = 50
		self.gaussian_method.fit_result_dict = \
			{'lambda_1': 7.75, 'mu_1': 103.7, 'sigma_1': 154.9, \
			'lambda_2': 4200, 'mu_2': -61096.85, 'sigma_2': 22622.0}
		mu_to_peak_distvec = self.gaussian_method._calc_mu_distance_to_peak()
		expected_threshold = 103.7 + 2*154.9
		test_threshold = \
			self.gaussian_method._find_threshold_with_distant_peaks(
				mu_to_peak_distvec)

	def test_threshold_based_on_mu_2(self):
		'''
		Calculate threshold based on mu, with the first distribution
		used for threshold calculation
		(Flip result values from test_threshold_based_on_mu_1)
		'''
		self.gaussian_method.peak_x_pos = 50
		self.gaussian_method.fit_result_dict = \
			{'lambda_2': 7.75, 'mu_2': 103.7, 'sigma_2': 154.9, \
			'lambda_1': 4200, 'mu_1': -61096.85, 'sigma_1': 22622.0}
		mu_to_peak_distvec = self.gaussian_method._calc_mu_distance_to_peak()
		expected_threshold = 103.7 + 2*154.9
		test_threshold = \
			self.gaussian_method._find_threshold_with_distant_peaks(
				mu_to_peak_distvec)

	def test_threshold_based_on_sigma_1_peak_close_to_0(self):
		'''
		Calculate threshold based on sigma, with the first distribution
		used for threshold calculation, due to the peak x position being
		too close to 0
		'''
		self.gaussian_method.peak_x_pos = 2
		self.gaussian_method.fit_result_dict = \
			{'lambda_1': 7.75, 'mu_1': -61096.85, 'sigma_1': 154.9, \
			'lambda_2': 4200, 'mu_2': 103.7, 'sigma_2': 22622.0}
		mu_to_peak_distvec = self.gaussian_method._calc_mu_distance_to_peak()
		expected_threshold = -61096.85 + 2*154.9
		test_threshold = \
			self.gaussian_method._find_threshold_with_distant_peaks(
				mu_to_peak_distvec)

	def test_threshold_based_on_sigma_2_mus_close_to_peak(self):
		'''
		Calculate threshold based on sigma, with the first distribution
		used for threshold calculation, due to the peak x position being
		too close to 0
		'''
		self.gaussian_method.peak_x_pos = 50
		self.gaussian_method.fit_result_dict = \
			{'lambda_1': 7.75, 'mu_1': 103.7, 'sigma_1': 500.0, \
			'lambda_2': 4200, 'mu_2': -1000, 'sigma_2': 154.9}
		mu_to_peak_distvec = self.gaussian_method._calc_mu_distance_to_peak()
		expected_threshold = -1000 + 2*154.9
		test_threshold = \
			self.gaussian_method._find_threshold_with_distant_peaks(
				mu_to_peak_distvec)
	
### unittests for _mu1PosThresholdMethod ###

class TestIDThreshold_mu1Pos(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.array_data = \
			np.loadtxt('PIE_tests/test_ims/tophat_im_small_best_hist.csv',
				delimiter=',')

	def setUp(self):
		self.mu1_pos_method = \
			_mu1PosThresholdMethod(self.array_data[0], self.array_data[2])
		# self.gaussian_method._min_real_peak_x_pos = 2.772380952380952
		# self.gaussian_threshold_standin._close_to_peak_dist = 2000

	def test_good_rsq_mu_2_neg(self):
		'''
		Test that when adjusted r squared is high and mu_2 is negative,
		threshold calculated as mu_1 + 2*sigma_1, despite low peak_x_pos
		'''
		self.mu1_pos_method.peak_x_pos = 1
		self.mu1_pos_method.rsq_adj = 0.999
		self.mu1_pos_method.fit_result_dict = \
			{'lambda_1': 7.75, 'mu_1': 103.7, 'sigma_1': 154.9, \
			'lambda_2': 4200, 'mu_2': -61096.85, 'sigma_2': 22622.0}
		expected_threshold = 103.7 + 2*154.9
		self.mu1_pos_method._id_threshold()
		# check method name, flag, and calculated threshold
		self.assertEqual('mu_1+2*sigma_1[mu_1-positive]',
			self.mu1_pos_method.method_name)
		self.assertEqual(0, self.mu1_pos_method.threshold_flag)
		assert_allclose(expected_threshold, self.mu1_pos_method.threshold)

	def test_good_rsq_mu_2_pos(self):
		'''
		Test that when adjusted r squared is high and mu_2 is positive,
		threshold calculated as mu + 2*sigma for mu closest to peak
		'''
		self.mu1_pos_method.peak_x_pos = 100
		self.mu1_pos_method.rsq_adj = 0.999
		self.mu1_pos_method.fit_result_dict = \
			{'lambda_2': 7.75, 'mu_2': 103.7, 'sigma_2': 154.9, \
			'lambda_1': 4200, 'mu_1': 5000, 'sigma_1': 10000.0}
		expected_threshold = 103.7 + 2*154.9
		self.mu1_pos_method._id_threshold()
		# check method name, flag, and calculated threshold
		self.assertEqual('mu_1+2*sigma_1[mu_1-positive]',
			self.mu1_pos_method.method_name)
		self.assertEqual(0, self.mu1_pos_method.threshold_flag)
		assert_allclose(expected_threshold, self.mu1_pos_method.threshold)

	def test_poor_rsq_correct_mu1_peak(self):
		'''
		Test that when adjusted r squares is low but mu_1 is close to
		the overall peak, threshold calculated as mu_1 + 2*sigma_1 but
		with threshold flag and poor fit method name
		'''
		self.mu1_pos_method.peak_x_pos = 100
		self.mu1_pos_method.y_peak_height = 10
		self.mu1_pos_method.rsq_adj = 0.001
		self.mu1_pos_method.fit_result_dict = \
			{'lambda_1': 7.75, 'mu_1': 103.7, 'sigma_1': 154.9, \
			'lambda_2': 4200, 'mu_2': -61096.85, 'sigma_2': 22622.0}
		expected_threshold = 103.7 + 2*154.9
		self.mu1_pos_method._id_threshold()
		# check method name, flag, and calculated threshold
		self.assertEqual('mu_1+2*sigma_1[mu_1-positive]_poor_minor_fit',
			self.mu1_pos_method.method_name)
		self.assertEqual(5, self.mu1_pos_method.threshold_flag)
		assert_allclose(expected_threshold, self.mu1_pos_method.threshold)

	def test_poor_rsq_tall_mu1_peak(self):
		'''
		Test that when adjusted r squares is low and first gaussian is
		too high, NaN is returned for threshold
		'''
		self.mu1_pos_method.peak_x_pos = 100
		self.mu1_pos_method.y_peak_height = 10
		self.mu1_pos_method.rsq_adj = 0.001
		self.mu1_pos_method.fit_result_dict = \
			{'lambda_1': 50, 'mu_1': 103.7, 'sigma_1': 154.9, \
			'lambda_2': 4200, 'mu_2': -61096.85, 'sigma_2': 22622.0}
		self.mu1_pos_method._id_threshold()
		# check threshold missing
		self.assertTrue(np.isnan(self.mu1_pos_method.threshold))

	def test_poor_rsq_distant_peaks(self):
		'''
		Test that when adjusted r squared is low and both gaussians
		have mu values far from the overall peak, NaN is returned for
		threshold
		'''
		self.mu1_pos_method.peak_x_pos = 100
		self.mu1_pos_method.y_peak_height = 10
		self.mu1_pos_method.rsq_adj = 0.001
		self.mu1_pos_method.fit_result_dict = \
			{'lambda_1': 7.75, 'mu_1': 200, 'sigma_1': 154.9, \
			'lambda_2': 4200, 'mu_2': -61096.85, 'sigma_2': 22622.0}
		self.mu1_pos_method._id_threshold()
		# check threshold missing
		self.assertTrue(np.isnan(self.mu1_pos_method.threshold))

### unittests for _mu1ReleasedThresholdMethod ###

class TestIDThreshold_mu1Released(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.array_data = \
			np.loadtxt('PIE_tests/test_ims/tophat_im_small_best_hist.csv',
				delimiter=',')

	def setUp(self):
		self.mu1_released_method = \
			_mu1ReleasedThresholdMethod(self.array_data[0], self.array_data[2])
		# self.gaussian_method._min_real_peak_x_pos = 2.772380952380952
		# self.gaussian_threshold_standin._close_to_peak_dist = 2000

	def test_both_mu_neg(self):
		'''
		Test that when both mu values are negative, NaN is returned for
		threshold
		'''
		self.mu1_released_method.peak_x_pos = 100
		self.mu1_released_method.rsq_adj = 0.999
		self.mu1_released_method.fit_result_dict = \
			{'lambda_2': 7.75, 'mu_2': -103.7, 'sigma_2': 154.9, \
			'lambda_1': 4200, 'mu_1': -5000, 'sigma_1': 10000.0}
		self.mu1_released_method._id_threshold()
		self.assertTrue(np.isnan(self.mu1_released_method.threshold))

	def test_poor_rsq(self):
		'''
		Test that when adjusted r squared is low, NaN is returned for
		threshold
		'''
		self.mu1_released_method.peak_x_pos = 100
		self.mu1_released_method.rsq_adj = 0.001
		self.mu1_released_method.fit_result_dict = \
			{'lambda_2': 7.75, 'mu_2': 103.7, 'sigma_2': 154.9, \
			'lambda_1': 4200, 'mu_1': 5000, 'sigma_1': 10000.0}
		self.mu1_released_method._id_threshold()
		self.assertTrue(np.isnan(self.mu1_released_method.threshold))

	def test_good_rsq_pos_mu(self):
		'''
		Test that when adjusted r squared is high and both mu values
		positive, returns mean based on
		_find_threshold_with_distant_peaks method
		'''
		self.mu1_released_method.peak_x_pos = 100
		self.mu1_released_method.rsq_adj = 0.999
		self.mu1_released_method.fit_result_dict = \
			{'lambda_2': 7.75, 'mu_2': 103.7, 'sigma_2': 154.9, \
			'lambda_1': 4200, 'mu_1': 5000, 'sigma_1': 10000.0}
		expected_threshold = 103.7 + 2*154.9
		self.mu1_released_method._id_threshold()
		# check method name, flag, and calculated threshold
		self.assertEqual('mu_1+2*sigma_1[mu_1-released]',
			self.mu1_released_method.method_name)
		self.assertEqual(2, self.mu1_released_method.threshold_flag)
		assert_allclose(expected_threshold, self.mu1_released_method.threshold)

	def test_one_pos_mu_close_to_correct_peak(self):
		'''
		Tests that when there is one positive mu close to the ovarall
		peak of the fitted distribution, and that peak approximates the
		legitimate peak of the background pixel values, threshold is
		chosen based on the positive peak
		'''
		self.mu1_released_method.peak_x_pos = 100
		self.mu1_released_method.rsq_adj = 0.999
		self.mu1_released_method.fit_result_dict = \
			{'lambda_2': 7.75, 'mu_2': 103.7, 'sigma_2': 154.9, \
			'lambda_1': 4200, 'mu_1': -5000, 'sigma_1': 10000.0}
		expected_threshold = 103.7 + 2*154.9
		self.mu1_released_method._id_threshold()
		# check method name, flag, and calculated threshold
		self.assertEqual('mu_1+2*sigma_1[mu_1-released]',
			self.mu1_released_method.method_name)
		self.assertEqual(2, self.mu1_released_method.threshold_flag)
		assert_allclose(expected_threshold, self.mu1_released_method.threshold)

	def test_one_pos_mu_close_to_incorrect_peak(self):
		'''
		Tests that when there is one positive mu close to the ovarall
		peak of the fitted distribution, but that peak doesn't
		approximate the legitimate peak of the background pixel values,
		NaN threshold is returned
		'''
		self.mu1_released_method.peak_x_pos = 1
		self.mu1_released_method.rsq_adj = 0.999
		self.mu1_released_method.fit_result_dict = \
			{'lambda_2': 7.75, 'mu_2': 103.7, 'sigma_2': 154.9, \
			'lambda_1': 4200, 'mu_1': -5000, 'sigma_1': 10000.0}
		self.mu1_released_method._id_threshold()
		self.assertTrue(np.isnan(self.mu1_released_method.threshold))

	def test_one_pos_mu_far_from_correct_peak(self):
		'''
		Tests that when there is one positive mu that isn't close to the
		ovarall peak of the fitted distribution, NaN threshold is
		returned
		'''
		self.mu1_released_method.peak_x_pos = 100
		self.mu1_released_method.rsq_adj = 0.999
		self.mu1_released_method.fit_result_dict = \
			{'lambda_2': 7.75, 'mu_2': 5000, 'sigma_2': 154.9, \
			'lambda_1': 4200, 'mu_1': -5000, 'sigma_1': 10000.0}
		self.mu1_released_method._id_threshold()
		self.assertTrue(np.isnan(self.mu1_released_method.threshold))

### unittests for _SlidingCircleThresholdMethod ###
class TestFindXStep(unittest.TestCase):

	def setUp(self):
		self.sliding_circle_standin = \
			object.__new__(_SlidingCircleThresholdMethod)

	def test_large_xstep(self):
		'''
		Tests xstep that is > 1
		'''
		element_num = 910
		xstep_multiplier = 0.03
		expected_xstep = 27
		test_xstep = \
			self.sliding_circle_standin._find_xstep(element_num, xstep_multiplier)
		self.assertEqual(expected_xstep, test_xstep)
		self.assertTrue(isinstance(test_xstep, np.integer) or isinstance(test_xstep, int))

	def test_small_xstep(self):
		'''
		Tests xstep that is < 1 (and thus returns 1)
		'''
		element_num = 910
		xstep_multiplier = 0.0001
		expected_xstep = 1
		test_xstep = \
			self.sliding_circle_standin._find_xstep(element_num, xstep_multiplier)
		self.assertEqual(expected_xstep, test_xstep)
		self.assertTrue(isinstance(test_xstep, np.integer) or isinstance(test_xstep, int))

class TestSampleandStrechGraph(unittest.TestCase):

	def setUp(self):
		self.sliding_circle_standin = \
			object.__new__(_SlidingCircleThresholdMethod)
		self.sliding_circle_standin._x_stretch_factor = 0.1
		self.sliding_circle_standin._y_stretch_factor = 100
		self.sliding_circle_standin.x_vals = np.array([100, 150, 250, 300, 500, 550, 700]).astype(float)
		self.sliding_circle_standin.y_vals = np.array([5, 10, 9, 7, 6, 5.5, 4])

	def test_stretch_and_subsample_step_1(self):
		'''
		Tests that when _xstep is 1, original arrays multiplied by
		stretch factors
		'''
		self.sliding_circle_standin._xstep = 1
		self.sliding_circle_standin._sample_and_stretch_graph()
		expected_x_stretched = self.sliding_circle_standin.x_vals/10
		expected_y_stretched = self.sliding_circle_standin.y_vals*100
		assert_array_equal(expected_x_stretched,
			self.sliding_circle_standin.x_vals_stretched)
		assert_array_equal(expected_y_stretched,
			self.sliding_circle_standin.y_vals_stretched)

	def test_stretch_and_subsample_step_3(self):
		'''
		Tests that when _xstep is 3, every 3rd value of original array
		(starting with position 0) is multiplied by the appropriate
		stretch factor and placed in stretch version of array
		'''
		self.sliding_circle_standin._xstep = 3
		self.sliding_circle_standin._sample_and_stretch_graph()
		expected_x_stretched = np.array([10, 30, 70])
		expected_y_stretched = np.array([500, 700, 400])
		assert_array_equal(expected_x_stretched,
			self.sliding_circle_standin.x_vals_stretched)
		assert_array_equal(expected_y_stretched,
			self.sliding_circle_standin.y_vals_stretched)

class TestCreatePolyMask(unittest.TestCase):

	def setUp(self):
		self.sliding_circle_standin = \
			object.__new__(_SlidingCircleThresholdMethod)

	def test_poly_mask(self):
		'''
		Test polygon mask image creation
		'''
		self.sliding_circle_standin.x_vals_stretched = \
			np.array([1, 2.9, 4, 5.9]).astype(float)
		self.sliding_circle_standin.y_vals_stretched = \
			np.array([2, 4, 1.9, 1]).astype(float)
		expected_mask = np.array([
			[1, 1, 1, 1, 1, 1],
			[0, 1, 1, 1, 1, 1],
			[0, 1, 1, 1, 0, 0],
			[0, 0, 1, 1, 0, 0]
			])
		self.sliding_circle_standin._create_poly_mask()
		assert_array_equal(expected_mask, self.sliding_circle_standin._fit_im)




if __name__ == '__main__':
	unittest.main()