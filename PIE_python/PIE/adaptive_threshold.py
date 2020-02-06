#!/usr/bin/python

'''
Performs automatic thresholding on imaging to identify cell centers
'''

import cv2
import numpy as np
import warnings
import sys
from scipy import signal
from scipy.optimize import least_squares

class _LogHistogramSmoother(object):
	'''
	Performs smoothing on log histogram
	'''
	def _set_smoothing_window_size(self, ln_tophat_hist, window_size):
		'''
		Sets the window size to be used for smoothing, which must be odd
		Window size should not be more than 1/3 the # of elements in the
		histogram (heuristic)
		'''
		hist_elements = len(ln_tophat_hist)
		if window_size > hist_elements/3:
			# round to the nearest odd number
			smooth_window_size = \
				2*int(round((float(hist_elements)/3+1)/2))-1
		else:
			smooth_window_size = window_size
		return(smooth_window_size)

	def _smooth_log_histogram(self, ln_tophat_hist, window_size):
		'''
		Performs Savitzky-Golay filtration on log of tophat histogram
		with 3rd degree polynomial order
		'''
		# set smoothing window size
		smooth_window_size = self._set_smoothing_window_size(ln_tophat_hist,
			window_size)
		ln_tophat_smooth = \
			signal.savgol_filter(ln_tophat_hist, smooth_window_size, 3)
		return(ln_tophat_smooth)

class _ThresholdFinder(_LogHistogramSmoother):
	'''
	Finds adaptive threshold for image
	'''
	def __init__(self, input_im):
		self.input_im = input_im
		# set tophat structural element to circle with radius 10
		# using the default ellipse struct el from cv2 gives a different
		# result than in matlab
		#self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
		# manually create matlab strel('disk', 10) here
		self.kernel = np.uint8([
			[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
			[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
			[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
			[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
			[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
			[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
			[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
			[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]])
			# TODO: This is a place that we can consider changing in future
			# versions: the radius here should really depend on expected
			# cell size and background properties (although some prelim
			# tests showed increasing element radius had only negative
			# consequences)
			# Also a good idea to see if the corresponding cv2
			# ellipse structuring element works here
		# set a warning flag to 0 (no warning)
		self.threshold_flag = 0
			# TODO: would be good to make an enum class for these
		self.default_smoothing_window_size = 21
			# TODO: heuristic - maybe better to use number of histogram
			# elements here?

	def _get_tophat(self):
		'''
		Gets tophat of an image
		'''
		self.tophat_im = \
			cv2.morphologyEx(self.input_im, cv2.MORPH_TOPHAT, self.kernel)

	def _get_unique_tophat_vals(self):
		'''
		Counts how many unique values there are in the tophat image
		Throws a warning if the number is less than/equal to 200
		Throws an error if the number is less than/equal to 3
		'''
		# !!! It's important to make sure this code is run with try-except in the pipeline to avoid an error for one image derailing the whole analysis
		tophat_unique = np.unique(self.tophat_im)
		if len(tophat_unique) <= 3:
			raise ValueError('3 or fewer unique values in tophat image')
		elif len(tophat_unique) <= 200:
			self.threshold_flag = 1
		return(tophat_unique)

	def _bin_centers_to_edges(self, bin_centers):
		'''
		Calculates edges of histogram bins given bin centers in the same
		way as matlab hist function: each internal edge is equidistant
		from the two bin centers around it, external edges are -Inf and
		Inf
		'''
		internal_edges = (bin_centers.astype(float)[0:-1] +
			bin_centers.astype(float)[1:])/2
		bin_edges = np.concatenate(([-np.inf], internal_edges, [np.inf]),
			axis = 0)
		return(bin_edges)

	def _reproduce_matlab_hist(self, x, bins):
		'''
		Reproduces behavior of matlab hist function
		x is an array, and bins is either an integer for the number of
		equally sized bins, or a vector with the centers of unequally
		spaced bins to be used
		Unable to reproduce matlab behavior for an array x with integer
		values on the edges of the bins because matlab behaves
		unpredicatably in those cases
		e.g.: self._reproduce_matlab_hist(np.array([0, 0, 2, 3, 0, 2]), 3)
		'''
		# if bins is a numpy array, treat it as a list of bin
		# centers, and convert to bin edges
		if isinstance(bins, np.ndarray):
			# identify bin edges based on centers as matlab hist
			# function would
			bin_centers = bins
			bin_edges = self._bin_centers_to_edges(bins)
			# get histogram using bin_edges
			(counts, _) = np.histogram(x, bin_edges)
		elif isinstance(bins, int):
			# get histogram using bins as number of equally sized bins
			(counts, bin_edges) = np.histogram(x, bins)
			# identify bin centers
			bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2
		else:
			raise TypeError('bins may be either an integer for the ' +
				'number of equally sized bins, or a vector with the centers ' +
				'of unequally spaced bins to be used')
		return(counts, bin_centers)

	def _get_log_tophat_hist(self, tophat_bins):
		'''
		Calculates the log of histogram values of tophat_im at bins
		tophat_bins is either an integer for the number of equally sized
		bins, or a vector with the centers of unequally spaced bins to
		be used
		Returns log of histogram and bin centers
		'''
		# calculate histogram of self.tophat_im, as in matlab
		tophat_hist, bin_centers = \
			self._reproduce_matlab_hist(self.tophat_im.flatten(), tophat_bins)
		# take logs of histogram y values
		# mask invalid entries (e.g. where tophat_hist == 0) and replace
		# them with 0s
		ln_tophat_hist = np.ma.log(tophat_hist).filled(0)
		return(ln_tophat_hist, bin_centers)

	def _autocorrelate(self, x):
		'''
		Performs autocorrelation on x
		Returns autocorrelation and the lags at which it was calculated
		'''
		sample_num = len(x)
		lags = np.arange(-(sample_num-1), sample_num)
		autocorrelation = np.correlate(x, x, mode = 'full')
		return(autocorrelation, lags)

	def _check_autocorrelation_peaks(self, autocorrelation,
		prop_freq_space = 0.02):
		'''
		Counts number of peaks in autocorrelation within prop_freq_space
		proportion of frequency space around 0
		If any peaks found besides the major peak at lag=0, return True
		Otherwise, return False
		Previous experience has shown that setting prop_freq_space to
		1/50 produces good results in terms of identifying when the
		histogram is estimated over too many points
		'''
		# find how many elements in freq space in one directon (past the
		# peak)
		unidirectional_freq_elements = (len(autocorrelation)-1)/2
		# identify index at which major (0th) peak can be found
		zero_position = unidirectional_freq_elements
		# identify how many autocorrelation elements past the major peak
		# to look for peaks in
		max_lag_elements_from_zero = \
			int(round(unidirectional_freq_elements * prop_freq_space))
		# identify section of autocorrelation to look for peaks in
		autocorrelation_section = \
			autocorrelation[
				(zero_position + 
					np.arange(0, max_lag_elements_from_zero)).astype(int)]
		# look for peaks as deviations from monotonic decrease
		peaks_present = np.any(np.diff(autocorrelation_section) > 0)
		return(peaks_present)

	def _identify_best_histogram(self):
		'''
		Identify the best (log) histogram of the tophat image to use for
		downstream fitting steps and threshold identification
		Involves trying a pre-determined number of bins (for large
		images, best bins are the values of unique tophat values, but
		smaller images need much fewer bins)
		Then look for peaks in autocorrelation of ln_tophat_hist, which
		are a sign that the number of histogram bins was poorly
		selected; if peaks present, reduce number of bins and try again
		'''
		# get unique tophat vals
		tophat_unique = self._get_unique_tophat_vals()
		# set max number of bins to be used for tophat histogram
		# the value below seems to work well for a default
		max_bin_num = int(round(float(self.tophat_im.size)/3000))
		unique_tophat_vals = len(tophat_unique)
		# loop through identifying log histograms until one passes
		# autocorrelation test
		while True:
			# if max_bin_num is higher than the number of unique tophat
			# values, use tophat_unique as bins; otherwise, use
			# max_bin_num equally spaced bins
			if max_bin_num > unique_tophat_vals:
				ln_tophat_hist, bin_centers = \
					self._get_log_tophat_hist(tophat_unique)
			else:
				ln_tophat_hist, bin_centers = \
					self._get_log_tophat_hist(max_bin_num)
			# measure autocorrelation of histogram
			hist_autocorrelation, _ = self._autocorrelate(ln_tophat_hist)
			# check whether peaks exist in autocorrelation
			autocorr_peaks_exist = \
				self._check_autocorrelation_peaks(hist_autocorrelation)
			if autocorr_peaks_exist:
				# reduce max_bin_num by setting it to either the number
				# of unique tophat values or 2/3 the current max_bin_num
				# (whichever is smaller); allow the loop to run again
				max_bin_num = \
					min(unique_tophat_vals, int(round(float(max_bin_num)*2/3)))
			else:
				self.ln_tophat_hist = ln_tophat_hist
				self.x_pos = bin_centers
				break

	def _perform_thresholding(self):
		'''
		Use smooth log histogram to find best threshold
		'''
		### !!! NEEDS UNITTEST
		# try thresholding using _mu1PosThresholdMethod
		self.threshold_method = \
			_mu1PosThresholdMethod(self.x_pos, self.ln_tophat_smooth)
		self.threshold_method.get_threshold()
		# if _mu1PosThresholdMethod returns NaN threshold, try
		# 'mu1ReleasedThresholdMethod
		if np.isnan(self.threshold_method.threshold):
			self.threshold_method = \
				_mu1ReleasedThresholdMethod(self.x_pos, self.ln_tophat_smooth)
			self.threshold_method.get_threshold()

	def threshold_image(self):
		# tophat transform on input image
		self._get_tophat()
		# identify the best (least periodically bumpy) histogram of the
		# image to use for identifying threshold
		self._identify_best_histogram()
		# smooth the log of the histogram values
		self.ln_tophat_smooth = self._smooth_log_histogram(self.ln_tophat_hist,
			self.default_smoothing_window_size)

class _ThresholdMethod(object):
	'''
	Generic class for methods for finding threshold
	(See _GaussianFitThresholdMethod and _RollingCircleThresholdMethod)
	'''

	def __init__(self, method_name, threshold_flag, x_vals, y_vals):
		self.method_name = method_name
		self.x = x_vals.astype(float)
		self.y = y_vals.astype(float)
		# initialize threshold flag at 0
		self.threshold_flag = threshold_flag

	def _perform_fit(self):
		'''
		Performs fitting procedure
		'''
		pass

	def _id_threshold(self):
		'''
		Identify threshold
		'''
		pass

	def get_threshold(self):
		'''
		Perform fit, calculate and return threshold
		'''
		self._perform_fit()
		self._id_threshold()

class _GaussianFitThresholdMethod(_ThresholdMethod):
	'''
	Generic class for methods for finding a threshold that involve
	fitting a mixture of two gaussians
	'''

	def __init__(self, method_name, threshold_flag, x_vals, y_vals,
		lower_bounds, upper_bounds):
		super(_GaussianFitThresholdMethod, self).__init__(
			method_name, threshold_flag, x_vals, y_vals)
		self.param_idx_dict = \
			{'lambda_1': 0, 'mu_1': 1, 'sigma_1': 2, 'lambda_2': 3, 'mu_2': 4,
				'sigma_2': 5}
		# the following parameters cannot have values below 0
		self.non_neg_params = ['lambda_1', 'lambda_2']
		# the following parameters must be above 0
		self.above_zero_params = ['sigma_1', 'sigma_2']
		self.lower_bounds = self._check_bounds(lower_bounds)
		self.upper_bounds = self._check_bounds(upper_bounds)
		### some heuristics related to determining thresholds from ###
		###             gaussian fits to log histogram             ###
		# specify the lowest x position at which a believeable
		# histogram peak can be found, rather than a peak resulting
		# from a pileup of 0s
		self._min_real_peak_x_pos = 0.0025 * np.max(self.x)
		# specify a distance at which the peak of a single gaussian
		# component is considered 'close' to the overall peak
		self._close_to_peak_dist = 0.05 * np.max(x_vals)
		# specify a distance in the y-axis within which the highest peak
		# of a gaussian can be considered a 'close enough' approximation
		# for the highest peak of the full histogram
		self._vertical_close_to_peak_dist = 20
			# TODO: This value is too high to be reasonable, and
			# anyways, it should be scaled to the log of the number of
			# pixels in the image
			# For now, leaving this at 20 to be consistent with current
			# matlab code + paper, but this effectively blocks off the
			# "sliding circle on fit" route
		# lower cutoff value for adjusted r squared to be considered a
		# good fit
		self.good_fit_rsq = 0.85

	def _check_bounds(self, bounds):
		'''
		Sets bounds on gaussian parameters
		Checks that bounds is a numpy array of the correct size
		For parameters that are required to be non-negative, sets any
		negative parameters to 0
		For parameters that are required to be above 0, sets any
		parameters below the min system float to that value
		'''
		min_allowed_number = sys.float_info.min
		if isinstance(bounds, np.ndarray) and \
			len(bounds) == len(self.param_idx_dict):
			for param in self.non_neg_params:
				if bounds[self.param_idx_dict[param]] < 0:
					warnings.warn(('Bound for {0} below 0; re-setting this ' +
					'value to 0').format(param), UserWarning)
					bounds[self.param_idx_dict[param]] = 0
			for param in self.above_zero_params:
				if bounds[self.param_idx_dict[param]] < min_allowed_number:
					warnings.warn(('Bound for {0} below {1}; re-setting this ' +
					'value to {1}').format(param, min_allowed_number),
					UserWarning)
					bounds[self.param_idx_dict[param]] = min_allowed_number
		else:
			raise TypeError('bounds must be a numpy array of length ' +
				len(self.param_idx_dict))
		return(bounds)

	def _id_starting_vals(self):
		'''
		Identifies starting values for parameters
		Totally heuristic; will not reproduce matlab's behavior
		'''
		### !!! NEED TO ADD CHECKING THAT STARTING PARAMS BETWEEN BOUNDS
		self.starting_param_vals = np.zeros(len(self.param_idx_dict))
		max_y = np.max(self.y)
		min_x = np.min(self.x)
		# identify highest x value corresponding to a y value of >5% of
		# the max of y
		# (effectively the distant tail of the distribution)
		max_x_above_thresh = self.x[self.y > max_y*0.05][-1]
		x_span = max_x_above_thresh-min_x
		# first mean is ~1/20 of the way to this x value
		self.starting_param_vals[self.param_idx_dict['mu_1']] = x_span/20
		# second mean is ~1/2 of the way to this x value
		self.starting_param_vals[self.param_idx_dict['mu_2']] = x_span/2
		# highest peak is highest value of y
		self.starting_param_vals[self.param_idx_dict['lambda_1']] = max_y
		# second highest peak is the value of y closest to mu_2
		mu_2_closest_idx = \
			(np.abs(self.x - self.starting_param_vals[
				self.param_idx_dict['mu_2']])).argmin()
		self.starting_param_vals[self.param_idx_dict['lambda_2']] = self.y[mu_2_closest_idx]
		self.starting_param_vals[self.param_idx_dict['sigma_1']] = x_span/12
		self.starting_param_vals[self.param_idx_dict['sigma_2']] = x_span/6

	def _digauss_calculator(self, x, lambda_1, mu_1, sigma_1, lambda_2, mu_2, sigma_2):
		'''
		Computes sum of two gaussians with weights lamdba_1 and
		lambda_2, respectively
		'''
		y = float(lambda_1)*np.exp(-((x-float(mu_1))/float(sigma_1))**2) + \
			float(lambda_2)*np.exp(-((x-float(mu_2))/float(sigma_2))**2)
		return(y)

	def _digaussian_residual_fun(self, params, x, y_data):
		'''
		Calculates residuals of difference between
		_digauss_calculator(x) and y_data 
		'''
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			y_params = \
				self._digauss_calculator(x,
					params[self.param_idx_dict['lambda_1']],
					params[self.param_idx_dict['mu_1']],
					params[self.param_idx_dict['sigma_1']],
					params[self.param_idx_dict['lambda_2']],
					params[self.param_idx_dict['mu_2']],
					params[self.param_idx_dict['sigma_2']])
		residuals = y_data - y_params
		return(residuals)

	def _fit_gaussians(self, starting_param_vals, x_vals, y_vals):
		'''
		Fits a mixture of two gaussians (whose combined probability
		doesn't necessarily sum to 1, since coefficients can be anything)
		'''
		self.fit_results = \
			least_squares(self._digaussian_residual_fun,
				starting_param_vals, args=(x_vals, y_vals),
				bounds = (self.lower_bounds, self.upper_bounds))
		self.y_hat = y_vals - self.fit_results.fun

	def _calc_fit_adj_rsq(self):
		'''
		Calculates adjusted r squared value for fit_results
		'''
		ss_tot = sum((self.y-np.mean(self.y))**2)
		ss_res = sum(self.fit_results.fun**2)
		# n is the number of points
		n = len(self.y)
		# p is the number of parameters
		p = len(self.fit_results.x)
		# this method does not match matlab behavior, which instead
		# (inexplicably) uses ss_res/(n-p) in the numerator
		self.rsq_adj = 1-(ss_res/(n-p-1))/(ss_tot/(n-1))

	def _find_peak(self):
		'''
		Finds highest point in mixture distribution, and its
		corresponding x value
		If two y values are equally high, returns x value corresponding
		to the first
		'''
		self.peak_x_pos = self.x[np.argmax(self.y_hat)]
		self.y_peak_height = np.max(self.y_hat)

	def _generate_fit_result_dict(self):
		'''
		Generates a dict of parameters that contains fit results
		'''
		self.fit_result_dict = {param_name: self.fit_results.x[param_idx]
			for param_name, param_idx in self.param_idx_dict.items()}

	def _calc_typical_threshold(self, gaussian_number):
		'''
		Calculates the most commonly used threshold, mean + 2*sd of one
		of the two best-fit gaussians (either 1 or 2, identified by
		gaussian_number)
		'''
		mean_param = 'mu_' + str(gaussian_number)
		sd_param = 'sigma_' + str(gaussian_number)
		threshold = \
			self.fit_result_dict[mean_param] + 2*self.fit_result_dict[sd_param]
		return(threshold)

	def _calc_mu_distance_to_peak(self):
		'''
		Calculates vector of distances of [mu_1, mu_2] to the peak x
		position of the fit
		'''
		mu_to_peak_distvec = \
			np.abs(self.peak_x_pos -
				np.array([self.fit_result_dict['mu_1'],
				self.fit_result_dict['mu_2']]))
		return(mu_to_peak_distvec)

	def _perform_fit(self):
		'''
		Performs fit with mixture of two gaussians and runs calculation
		of adjusted r squared and peak of distribution mixture
		'''
		# no unittest needed here
		self._id_starting_vals()
		self._fit_gaussians(self.starting_param_vals, self.x, self.y)
		self._calc_fit_adj_rsq()
		self._find_peak()

	def _find_threshold_with_distant_peaks(self, mu_to_peak_distvec,
		check_peak_pos = True):
		'''
		Calculates threshold for cases when peak x position not all the
		way at the lower side of the distribution, and at least one
		fitted gaussian's mu is far from the peak of the mixture
		distribution
		'''
		# check the x pos for highest point in y
		# find out which b is close to this x pos
		threshold_1 = self._calc_typical_threshold(1)
		threshold_2 = self._calc_typical_threshold(2)
		threshold_vec = np.array([threshold_1, threshold_2])
		# only check that peak position far enough from 0 if
		# check_peak_pos is True
		if check_peak_pos:
			peak_pos_ok = self.peak_x_pos > self._min_real_peak_x_pos
		else:
			peak_pos_ok = True
		if np.sum(mu_to_peak_distvec > self._close_to_peak_dist) > 0 and \
			peak_pos_ok:
			# use the distribution with the closest mu to the overall
			# peak to calculate the threshold if they are not
			# both very close to the peak and the peak is not followed
			# by a ditch and second peak at the very beginning
			threshold = threshold_vec[np.argmin(mu_to_peak_distvec)]
		else:
			# both mus are very close to the peak or the peak is
			# around 0, followed by a 'ditch' at the very beginning
			# in this case the peak position may not be a good standard
			# check sigmas - smaller sigma corresponds to the main peak
			sigma_vec = np.array([self.fit_result_dict['sigma_1'],
				self.fit_result_dict['sigma_2']])
			threshold = threshold_vec[np.argmin(sigma_vec)]
		return(threshold)

		
class _mu1PosThresholdMethod(_GaussianFitThresholdMethod):
	### !!! NEEDS BETTER METHOD DESCRIPTION

	def __init__(self, x_vals, y_vals):
		method_name = 'mu_1+2*sigma_1[mu_1-positive]'
		lower_bounds = np.array(
			[1, 0, sys.float_info.min, 0.5, -np.inf, sys.float_info.min])
		upper_bounds = np.array([np.inf]*6)
		threshold_flag = 0
		super(_mu1PosThresholdMethod, self).__init__(
			method_name, threshold_flag, x_vals, y_vals, lower_bounds,
			upper_bounds)

	def _id_threshold(self):
		'''
		Identify threshold
		'''
		mu_to_peak_distvec = self._calc_mu_distance_to_peak()
		if self.rsq_adj > self.good_fit_rsq:
			# mu_1 must be positive based on bounds
			# mu_2 may be negative
			if self.fit_result_dict['mu_2'] <= 0:
				self.threshold = self._calc_typical_threshold(1)
			else:
				# this method needs to know whether overall fit peak is
				# sufficiently far from 0
				self.threshold = \
					self._find_threshold_with_distant_peaks(mu_to_peak_distvec)
		elif np.abs(self.fit_result_dict['lambda_1']-self.y_peak_height) <= \
			self._vertical_close_to_peak_dist and \
			mu_to_peak_distvec[0] < self._close_to_peak_dist:
			# poor r sq because the smaller peak fit poor
			# but as long as the major peak fit good, as est by the cond here
			# should go ahead with b1+2*c1
			self.method_name = 'mu_1+2*sigma_1[mu_1-positive]_poor_minor_fit'
			self.threshold_flag = 5;
			self.threshold = self._calc_typical_threshold(1)
		else:
			self.threshold = np.nan

class _mu1ReleasedThresholdMethod(_GaussianFitThresholdMethod):
	### !!! NEEDS BETTER METHOD DESCRIPTION

	def __init__(self, x_vals, y_vals):
		method_name = 'mu_1+2*sigma_1[mu_1-released]'
		lower_bounds = np.array(
			[1, -np.inf, sys.float_info.min, 0.5, -np.inf, sys.float_info.min])
		upper_bounds = np.array([np.inf]*6)
		threshold_flag = 2
		super(_mu1ReleasedThresholdMethod, self).__init__(
			method_name, threshold_flag, x_vals, y_vals, lower_bounds,
			upper_bounds)

	def _id_threshold(self):
		'''
		Identify threshold
		'''
		mu_to_peak_distvec = self._calc_mu_distance_to_peak()
		if self.rsq_adj > self.good_fit_rsq and \
			(self.fit_result_dict['mu_1'] > 0 or \
				self.fit_result_dict['mu_2'] > 0):
			# if fit is good and both means positive, calculate
			# threshold based on b1+2*c1 same way as in
			# _mu1PosThresholdMethod
			if self.fit_result_dict['mu_1'] > 0 and \
				self.fit_result_dict['mu_2'] > 0:
				# no peak position requirement here, so set
				# check_peak_pos to False
				self.threshold = \
					self._find_threshold_with_distant_peaks(mu_to_peak_distvec,
						check_peak_pos = False)
			else:
				# either mu_1 < 0 or mu_2 < 0 (but not both)
				# if gaussian with the positive mean corresponds to
				# legitimate (distant from 0) peak of log histogram, it
				# can be used for threshold
				# otherwise, the estimate of sigma may be off (esp. if
				# peak of fitted distribution is the result of a pileup
				# of values close to 0 rather than the real background
				# vals of the image), causing problems using mu+2*sigma
				highest_mu_idx = np.argmax([self.fit_result_dict['mu_1'],
					self.fit_result_dict['mu_2']])
				if self.peak_x_pos > self._min_real_peak_x_pos and \
					mu_to_peak_distvec[highest_mu_idx] < \
						self._close_to_peak_dist:
					# mu values indexed starting at 1
					correct_distribution = highest_mu_idx + 1
					self.threshold = \
						self._calc_typical_threshold(correct_distribution)
				else:
					self.threshold = np.nan
		else:
			# if fit is bad or both mu vals are negative, return NaN
			# threshold
			self.threshold = np.nan

class _SlidingCircleThresholdMethod(_ThresholdMethod):
	'''
	Generic class for methods that involve finding the threshold by
	finding the point at which the highest proportion of a circle
	centered on the graph of the log histogram of tophat intensities is
	below the line
	'''
	def __init__(self, method_name, threshold_flag, x_vals, y_vals,
		xstep):
		super(_SlidingCircleThresholdMethod, self).__init__(
			method_name, threshold_flag, x_vals, y_vals)
		### some heuristics related to determining thresholds from ###
		###           sliding circle along log histogram           ###
		# specify the bounds on x positions between which sliding circle
		# operates (these are essentially bounds on where the threshold
		# may be found)
		self._lower_bound = 0.13 * np.max(self.x)
		self._upper_bound = 0.53 * np.max(self.x)
		# specify radius, in number of pixels (i.e. number of points
		# along x-axis)
		self._radius = 100
		# only include every xstep-th value in sliding circle
		# the idea here is to save time by not including nearly
		# identical values
		self._xstep = xstep
			# TODO: It might be good to set this based on autocorr
			# in the future, i.e. find distance at which
			# autocorrelation of ln histogram falls off by a certain
			# amount and use that

	def _find_xstep(self, element_num, xstep_multiplier):
		'''
		Finds xstep given xstep_multiplier
		Minimum possible value returned is 1
		'''
		xstep = np.max([1, np.floor(element_num * xstep_multiplier)]
			).astype(int)
		return(xstep)

	def _perform_fit(self):
		'''
		Performs fitting procedure
		'''
		pass

	def _id_threshold(self):
		'''
		Identify threshold
		'''
		pass

class _DataSlidingCircleThresholdMethod(_SlidingCircleThresholdMethod,
	_LogHistogramSmoother):
	'''
	Threshold method that takes in raw log histogram of tophat image,
	performs smoothing, and then finds threshold using sliding circle
	method
	'''
	def __init__(self, x_vals, raw_y_vals):
		threshold_flag = 4
		method_name = 'sliding_circle_data'
		xstep = self._find_xstep(len(x_vals), 0.003)
			# heuristic - see TODO in parent class
			# NB: current implementation will not exactly reproduce
				# matlab code, which hard-codes the xstep as either 3 or
				# 100, for data vs fit-based sliding circle, respectively
		self.default_smoothing_window_size = 57
			# TODO: heuristic - maybe better to use number of histogram
			# elements here?
		y_vals = self._smooth_log_histogram(raw_y_vals,
			self.default_smoothing_window_size)
		super(_DataSlidingCircleThresholdMethod, self).__init__(
			method_name, threshold_flag, x_vals, y_vals, xstep)

class _FitSlidingCircleThresholdMethod(_SlidingCircleThresholdMethod):
	'''
	Threshold method that takes in best fit to histogram of tophat
	image and finds threshold using sliding circle method
	'''
	def __init__(self, x_vals, y_vals):
		threshold_flag = 3
		method_name = 'sliding_circle_data'
		xstep = self._find_xstep(len(x_vals), 0.1)
			# heuristic - see TODO in parent class
			# NB: current implementation will not exactly reproduce
				# matlab code, which hard-codes the xstep as either 3 or
				# 100, for data vs fit-based sliding circle, respectively
		super(_FitSlidingCircleThresholdMethod, self).__init__(
			method_name, threshold_flag, x_vals, y_vals, xstep)

if __name__ == '__main__':

	pass
	# need to read in image file name via argparse, load image, pass it to threshold_finder, get back threshold mask

