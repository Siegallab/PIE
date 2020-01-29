#!/usr/bin/python

'''
Performs colony edge detection
'''

import cv2
import numpy as np
import warnings
import sys
from scipy import signal
from scipy.optimize import least_squares
from scipy.stats import norm
from sklearn import mixture

class _EdgeDetector(object):
	'''
	Detects edges in input_im using PIE algorithm
	'''

	def __init__(self, input_im_path):
		'''
		Read input_im_path as grayscale image of arbitrary bitdepth
		'''
		self.input_im = cv2.imread(input_im_path, cv2.IMREAD_ANYDEPTH)
		# create 'pie' quadrants
		self._get_pie_pieces()

	def _get_pie_pieces(self):
		'''
		By mutliplying x- by y- gradient directions, we get 4 sets if images,
		which, together, make up every pixel in the image; conveniently, each
		cell is represented as 4 separate quarters of a pie, each of which is
		surrounded by nothing.
		'''
		# Find x and y gradients
		# BORDER_REPLICATE makes Sobel filter behave like the
		# imgradientxy one in matlab at the image's borders
		Gx = cv2.Sobel(self.input_im, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REPLICATE)
		Gy = cv2.Sobel(self.input_im, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REPLICATE)
		# create gradient masks
		Gx_right = Gx < 0
		Gx_left = Gx > 0
		Gy_top = Gy > 0
		Gy_bottom = Gy < 0
		self.pie_piece_dict = dict()
		self.pie_piece_dict['i'] = _PiePiece(Gx_right, Gy_top)
		self.pie_piece_dict['ii'] = _PiePiece(Gx_left, Gy_top)
		self.pie_piece_dict['iii'] = _PiePiece(Gx_left, Gy_bottom)
		self.pie_piece_dict['iv'] = _PiePiece(Gx_right, Gy_bottom)

	def _create_inital_overlay(self):
		'''
		Runs edge detection without cleanup
		'''
		### ??? NEEDS UNITTEST ??? ###
		# identify cell centers with adaptive thresholding of tophat im
		cell_centers = _ThresholdFinder.threshold_image(self.input_im)
		# identify 'pie pieces' that overlap cell centers, and combine
		# them into a composite overlay
		initial_overlay = np.zeros(self.input_im.shape, dtype = bool)
		for pie_quad_name, pie_piece_quadrant in \
			self.pie_piece_dict.iteritems():
			current_cell_pieces = \
				pie_piece_quadrant.id_center_overlapping_pie_pieces(
					cell_centers)
			initial_overay = np.logical_or(initial_overay, current_cell_pieces)
		return(initial_overay)


class _PiePiece(object):
	'''
	Stores and operates on an individual 'pie piece'
	'''

	def __init__(self, x_grad_mask, y_grad_mask):
		'''
		Sets up a mask containing 'pie pieces' which correspond to a
		single unique quadrant of x and y gradient directions, where
		x_grad_mask and y_grad_mask are both true
		'''
		self.pie_mask = np.logical_and(x_grad_mask, y_grad_mask)

	def id_center_overlapping_pie_pieces(self, cell_center_mask):
		'''
		We identify the parts of self.pie_mask that belong to real
		cells by seeing which ones overlap with cell_center_mask
		'''
		# Label each contiguous object in the image with a unique int
		# When identifying the objects, we must not count diagonal
		# neighbor pixels as belonging to the same object (this is VERY
		# important)
		# Default for cv2.connectedComponents is connectivity = 8
		label_num, labeled_pie_mask = \
			cv2.connectedComponents(np.uint8(self.pie_mask), connectivity = 4)
		# Identify the positions in each labeled_objects that overlap
		# with cell_center_mask
		center_pie_overlap = labeled_pie_mask[cell_center_mask]
		# Identify the label numbers (allowed_objects) that overlap with
		# cell_center_mask
		allowed_labels_with_backgrnd = np.unique(center_pie_overlap)
		# excluding the background (where labeled_objects==0) is key!
		allowed_labels = \
			allowed_labels_with_backgrnd[allowed_labels_with_backgrnd != 0]
		# identify pixels in labeled_pie_mask whose labels are in
		# allowed_labels
		cell_overlapping_pie_mask = \
			np.isin(labeled_pie_mask, allowed_labels)
		return(cell_overlapping_pie_mask)

class _ThresholdFinder(object):
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
			# !!! This is a place that we can consider changing in future
			# versions: the radius here should really depend on expected
			# cell size and background properties (although some prelim
			# tests showed increasing element radius had only negative
			# consequences)
			# !!! Also a good idea to see if the corresponding cv2
			# ellipse structuring element works here
		# set a warning flag to 0 (no warning)
		self.threshold_flag = 0
			# !!! would be good to make an enum class for these

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
			raise(ValueError, '3 or fewer unique values in tophat image')
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
				(zero_position + np.arange(0, max_lag_elements_from_zero))]
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

	def _set_smoothing_window_size(self, default_window_size = 21):
		'''
		Sets the window size to be used for smoothing, which must be odd
		Window size should not be more than 1/3 the # of elements in the
		histogram (heuristic)
		'''
		hist_elements = len(self.ln_tophat_hist)
		if default_window_size > hist_elements/3:
			# round to the nearest odd number
			smooth_window_size = \
				2*int(round((float(hist_elements)/3+1)/2))-1
		else:
			smooth_window_size = default_window_size
		return(smooth_window_size)

	def _smooth_log_histogram(self):
		'''
		Performs Savitzky-Golay filtration on log of tophat histogram
		with 3rd degree polynomial order
		'''
		# set smoothing window size
		smooth_window_size = self._set_smoothing_window_size()
		self.ln_tophat_smooth = \
			signal.savgol_filter(self.ln_tophat_hist, smooth_window_size, 3)

	def threshold_image(self):
		# tophat transform on input image
		self._get_tophat()
		# identify the best (least periodically bumpy) histogram of the
		# image to use for identifying threshold
		self._identify_best_histogram()
		# smooth the log of the histogram values
		self._smooth_log_histogram()

class _ThresholdMethod(object):
	'''
	Generic class for methods for finding threshold
	(See _GaussianFitThresholdMethod and _RollingCircleThresholdMethod)
	'''

	def __init__(self, method_name, x_vals, y_vals):
		self.method_name = method_name
		self.x = x_vals.astype(float)
		self.y = y_vals.astype(float)

	def find_threshold(self):
		'''
		Identifies and returns threshold based on data
		'''
		pass

class _GaussianFitThresholdMethod(_ThresholdMethod):

	def __init__(self, method_name, x_vals, y_vals, lower_bounds, upper_bounds):
		super(_GaussianFitThresholdMethod, self).__init__(
			method_name, x_vals, y_vals)
		self.param_idx_dict = \
			{'lambda_1': 0, 'mu_1': 1, 'sigma_1': 2, 'lambda_2': 3, 'mu_2': 4,
				'sigma_2': 5}
		# the following parameters cannot have values below 0
		self.non_neg_params = ['lambda_1', 'lambda_2']
		# the following parameters must be above 0
		self.above_zero_params = ['sigma_1', 'sigma_2']
		self.lower_bounds = self._check_bounds(lower_bounds)
		self.upper_bounds = self._check_bounds(upper_bounds)

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

	def _calc_fit_adj_rsq(self):
		'''
		Calculates adjusted r squared value for fit_results
		'''
		ss_tot = sum((self.y-np.mean(self.y))**2)
		ss_res = sum(self.fit_results.fun**2)
#		rsq = 1-ss_res/ss_tot
		# n is the number of points
		n = len(self.y)
		# p is the number of parameters
		p = len(self.fit_results.x)
		# this method does not match matlab behavior, which instead
		# (inexplicably) uses ss_res/(n-p) in the numerator
		self.rsq_adj = 1-(ss_res/(n-p-1))/(ss_tot/(n-1))
		


				
	


if __name__ == '__main__':
	# !!! It's important to make sure this code is run with try-except in the pipeline to avoid an error for one image derailing the whole analysis (see _ThresholdFinder._get_unique_tophat_vals)

	input_im = cv2.imread('/Users/plavskin/Documents/yeast stuff/pie_paper_v2/plotcode/Fig2-5_ims/input_ims/xy01_12ms_3702.tif',
		cv2.IMREAD_ANYDEPTH)
	input_im_2 = input_im*32
	print(np.amax(input_im))
	print(np.amax(input_im_2))
	print(input_im.shape)
	cv2.imshow('Input Image', input_im_2)
	cv2.waitKey(10000)
	cv2.destroyAllWindows()