#!/usr/bin/python

'''
Performs automatic thresholding on imaging to identify cell centers
'''

import cv2
import numpy as np
import warnings
import sys
import pandas as pd
from PIE import ported_matlab
from PIE.density_fit import DensityFitterLS
from PIL import Image, ImageDraw
from plotnine import *
from scipy import signal

### TODO: Get rid of threshold flags (depracated)

# specify the multiplier by which s.d. of major peak-fitting
# density is multiplied to get threshold
# set sd_multiplier as a global parameter
sd_multiplier = 2

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

class _ThresholdMethodSelector(object):
	'''
	Returns threshold method object to use
	'''
	def __init__(self, cell_intensity_num, manual_method_order_list = None):
		if manual_method_order_list is not None:
			self.method_order = self._check_manual_method_order_list(
				manual_method_order_list
				)
		else:
			if cell_intensity_num == 1:
				self.method_order = [
					'mu1PosThresholdMethodTwoGauss',
					'mu1ReleasedThresholdMethod',
					'sliding_circle_selector'
					]
			elif cell_intensity_num == 2:
				self.method_order = [
					'mu1PosThresholdMethodThreeGauss',
					'mu1ReleasedThresholdMethod',
					'sliding_circle_selector'
					]
			else:
				raise ValueError('cell_intensity_num must be 1 or 2')

	def _check_manual_method_order_list(self, manual_method_order_list):
		'''
		Check that elements of manual_method_order_list are unique and 
		in the list of allowed threshold methods
		'''
		allowed_method_orders = set([
			'mu1PosThresholdMethodTwoGauss',
			'mu1PosThresholdMethodThreeGauss',
			'mu1ReleasedThresholdMethod',
			'sliding_circle_selector'
			])
		manual_method_order_set = set(manual_method_order_list)
		# check that values are unique
		if not len(manual_method_order_set) == len(manual_method_order_list):
			raise ValueError(
				'All values in manual_method_order_list must be unique'
				)
		elif not manual_method_order_set.issubset(allowed_method_orders):
			disallowed_methods = \
				list(manual_method_order_set-allowed_method_orders)
			raise ValueError(
				'manual_method_order_list contains the following '
				'disallowed elements: ' + str(disallowed_methods)
				)
		return(manual_method_order_list)

	def _sliding_circle_selector(self, threshold_method, x_pos, ln_tophat_hist):
		'''
		Factory method to return object of appropriate
		_SlidingCircleThresholdMethod-inheriting class
		'''
		### !!! NEEDS UNITTEST
		if hasattr(threshold_method, 'rsq_adj') and \
			threshold_method.rsq_adj > threshold_method.good_fit_rsq:
			threshold_method = \
				_FitSlidingCircleThresholdMethod(
					x_pos,
					threshold_method.data.y,
					threshold_method.data.y_hat
					)
		else:
			threshold_method = \
				_DataSlidingCircleThresholdMethod(
					x_pos, ln_tophat_hist
					)
		return(threshold_method)

	def get_threshold_method(self, x_pos, ln_tophat_hist, ln_tophat_smooth):
		'''Get threshold_method to use for thresholding'''
		### !!! NEEDS UNITTEST
		# initialize threshold_method as None
		# (in case first method is 'sliding_circle_selector')
		threshold_method = None
		# Try thresholding with each method in the order given in
		# self.method_order
		# While threshold_method returns np.nan threshold, go on to the
		# next method
		for method_key in self.method_order:
			# create threshold_method object
			# slightly messy, since _sliding_circle_selector takes 
			# different arguments than other threshold_method creators
			if method_key == 'sliding_circle_selector':
				threshold_method = self._sliding_circle_selector(
					threshold_method, x_pos, ln_tophat_hist
					)
			else:
				# get threshold_method by calling class named 
				# _<method_key>
				threshold_method = \
					globals()['_'+method_key](x_pos, ln_tophat_smooth)
			threshold = threshold_method.get_threshold()
			# only continue loop if threshold is np.nan
			if ~np.isnan(threshold):
				break
#		print(threshold_method.method_name)
#		print(threshold_method.rsq_adj)
		# return threshold method
		return(threshold_method)

class _ThresholdFinder(_LogHistogramSmoother):
	'''
	Finds adaptive threshold for image
	'''
	def __init__(
		self,
		input_im,
		image_type,
		cell_intensity_num, 
		manual_method_order_list = None
		):
		# treat input_im
		if image_type == 'bright':
			self.input_im = input_im
		elif image_type == 'dark':
			# the image needs to be inverted before thresholding,
			# then normalized so that assumptions about
			# peak values being close to 0 are true
			unnorm_input_im = cv2.bitwise_not(input_im)
			self.input_im = cv2.normalize(
				unnorm_input_im, None, alpha=0, beta=(2**16-1),
				norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U
				)
		else:
			raise ValueError(
				"image_type must be either 'bright' (bright cells on "
				"darker background) or 'dark' (dark cells on brighter "
				"background)"
				)
		self.threshold_method_selector = \
			_ThresholdMethodSelector(cell_intensity_num, manual_method_order_list)
		# set tophat structural element to circle with radius 10
		# using the default ellipse struct el from cv2 gives a different
		# result than in matlab
		#self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
		# manually create matlab strel('disk', 10) here
#		self.kernel = np.uint8([
#			[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
#			[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
#			[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#			[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#			[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#			[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#			[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#			[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
#			[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]])
		self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(24,24))
			# TODO: This is a place that we can consider changing in future
			# versions: the radius here should really depend on expected
			# cell size and background properties (although some prelim
			# tests showed increasing element radius had only negative
			# consequences)
			# It seems for our test images, ideal ellipse size is 
			# ~24x24, which is 2-3x the size of a single cell
		# set a warning flag to 0 (no warning)
		self.threshold_flag = 0
			# TODO: would be good to make an enum class for these
		self.default_smoothing_window_size = 21
			# TODO: heuristic - maybe better to use number of histogram
			# elements here?
		# set 'default' threshold method name
		self.default_threshold_method_class = \
			'_'+self.threshold_method_selector.method_order[0]

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
		tophat_unique = np.unique(self.tophat_im)
		if len(tophat_unique) <= 3:
			raise ValueError('3 or fewer unique values in tophat image')
		elif len(tophat_unique) <= 200:
			self.threshold_flag = 1
		return(tophat_unique)

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
			ported_matlab.hist(self.tophat_im.flatten(), tophat_bins)
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
		# the heuristic below seems to work well for a default
		max_bin_num = max(20, int(round(float(self.tophat_im.size)/3000)))
		unique_tophat_vals = len(tophat_unique)
		# if max_bin_num is higher than the number of unique tophat
		# values, use tophat_unique as bins; otherwise, use equally
		# spaced bins, with the bin number adjusted to avoid 
		# strong periodic elements
		# NB: Regularly spaced bins are not ideal, since they introduce 
		# artifacts, which is why we avoid using them if possible
		# However, likely to have too many possible unique_tophat_vals
		# in jpgs (and some tiffs as well)
		if max_bin_num > unique_tophat_vals:
			ln_tophat_hist, bin_centers = \
				self._get_log_tophat_hist(tophat_unique)
		else:
			# loop through identifying log histograms until one passes
			# autocorrelation test
			bin_num = max_bin_num
			while True:
				ln_tophat_hist, bin_centers = \
					self._get_log_tophat_hist(bin_num)
				# measure autocorrelation of histogram
				hist_autocorrelation, _ = self._autocorrelate(ln_tophat_hist)
				# check whether peaks exist in autocorrelation
				autocorr_peaks_exist = \
					self._check_autocorrelation_peaks(hist_autocorrelation)
				if autocorr_peaks_exist:
					# reduce max_bin_num by setting it to either the number
					# of unique tophat values or 2/3 the current max_bin_num
					# (whichever is smaller); allow the loop to run again
					bin_num = \
						min(unique_tophat_vals, int(round(float(bin_num)*2/3)))
				else:
					break
		self.ln_tophat_hist = ln_tophat_hist
		self.x_pos = bin_centers

	def _select_threshold(self):
		'''
		Use log histogram (or smooth log histogram) to find the optimal
		threshold
		'''
		### !!! NEEDS UNITTEST
		self.threshold_method = \
			self.threshold_method_selector.get_threshold_method(
				self.x_pos, self.ln_tophat_hist, self.ln_tophat_smooth
				)
	
	def _perform_thresholding(self, tophat_im, threshold):
		'''
		Thresholds tophat_im based on the threshold in
		self.threshold_method
		'''
		### !!! NEEDS UNITTEST
		# create a mask of 0s and 1s for input_im based on threshold
		_, threshold_mask = cv2.threshold(tophat_im, threshold, 1, cv2.THRESH_BINARY)
		# return mask as bool
		return(threshold_mask.astype(bool))

	def get_threshold_mask(self):
		# tophat transform on input image
		self._get_tophat()
		# identify the best (least periodically bumpy) histogram of the
		# image to use for identifying threshold
		self._identify_best_histogram()
		# smooth the log of the histogram values
		self.ln_tophat_smooth = self._smooth_log_histogram(self.ln_tophat_hist,
			self.default_smoothing_window_size)
		# run through threshold methods to select optimal one
		self._select_threshold()
		# threshold image
		self.threshold_mask = \
			self._perform_thresholding(self.tophat_im, self.threshold_method.threshold)
		return(self.threshold_mask)

class _ThresholdMethod(object):
	'''
	Generic class for methods for finding threshold
	(See _GaussianFitThresholdMethod and _SlidingCircleThresholdMethod)
	'''

	def __init__(self, method_name, threshold_flag, x_vals, y_vals):
		self.method_name = method_name
		self.data = pd.DataFrame({
			'x': x_vals.astype(float),
			'y': y_vals.astype(float)
			})
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

	def _create_ggplot(self, df, color_dict):
		'''
		Create ggplot-like plot of values in df, which must contain 'x',
		'y', 'data_type', 'linetype', and 'id' columns
		'''
		p = ggplot(df) + \
			geom_line(aes(x = 'x', y = 'y', color = 'data_type',
				linetype = 'linetype', group = 'id'), size = 1) + \
			geom_vline(xintercept = self.threshold,
				color='#984ea3', size=0.7) + \
			scale_x_continuous(name = 'pixel intensity') + \
			scale_y_continuous(name = 'log(count)') + \
			scale_color_manual(values = color_dict) + \
			scale_linetype_manual(values = \
				{'dashed':'dashed', 'solid':'solid'},
				guide = False) + \
			theme(legend_position = (0.75, 0.7),
				plot_title = element_text(face='bold'),
                panel_background = element_rect(fill='white'),
                panel_grid_major=element_line(color='grey',size=0.3),
                axis_line = element_line(color="black", size = 0.5),
                legend_title=element_blank(),
                legend_key = element_rect(fill='white'),
                legend_text=element_text(size=18,face='bold'),
                axis_text_x=element_text(size=18,face='bold'),
                axis_text_y=element_text(size=18,face='bold'),
                axis_title_x=element_text(size=20,face='bold'),
                axis_title_y=element_text(size=20,face='bold',angle=90)
				)
		return(p)

	def plot(self):
		'''
		Plot threshold identification graph
		'''
		pass

	def get_threshold(self):
		'''
		Perform fit, calculate and return threshold
		'''
		self._perform_fit()
		self._id_threshold()
		return(self.threshold)

class _GaussianFitThresholdMethod(_ThresholdMethod, DensityFitterLS):
	'''
	Generic class for methods for finding a threshold that involve
	fitting a mixture of gaussians
	'''

	def __init__(self, method_name, threshold_flag, x_vals, y_vals,
		lower_bounds, upper_bounds, dist_num, sd_multiplier_val):
		super(_GaussianFitThresholdMethod, self).__init__(
			method_name, threshold_flag, x_vals, y_vals)
		self.sd_multiplier = sd_multiplier_val
		self.dist_num_list = list(range(1,dist_num+1))
		# the following parameters cannot have values below 0
		self.non_neg_params = ['lambda_'+str(dist_count) for dist_count in self.dist_num_list]
		# the following parameters must be above 0
		self.above_zero_params = ['sigma_'+str(dist_count) for dist_count in self.dist_num_list]
		self.lower_bounds = self._check_bounds(lower_bounds)
		self.upper_bounds = self._check_bounds(upper_bounds)
		# set up some important values to be used in starting val calc
		self.max_y = np.max(self.data['y'])
		min_x = np.min(self.data['x'])
		# identify highest x value corresponding to a y value of >5% of
		# the max of y
		# (effectively the distant tail of the distribution)
		self.max_x_above_thresh = \
			self.data['x'].to_numpy()[self.data['y'] > self.max_y*0.05][-1]
		self.x_span = self.max_x_above_thresh-min_x
		### some heuristics related to determining thresholds from ###
		###             gaussian fits to log histogram             ###
		# specify the lowest x position at which a believeable
		# histogram peak can be found, rather than a peak resulting
		# from a pileup of 0s
		self._min_real_peak_x_pos = 0.0025 * np.max(self.data['x'])
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

	def _id_starting_vals(self):
		'''
		Identifies starting values for parameters

		Creates self.starting_param_vals, a numpy array of starting
		values with indices corresponding to those in
		self.param_idx_dict
		'''
		pass

	def _single_gauss_calculator(self, x, l, mu, sigma):
		'''
		Computes a single gaussian with weight l
		'''
		y = float(l)*np.exp(-((x-float(mu))/float(sigma))**2)
		return(y)

	def _calc_fit_adj_rsq(self):
		'''
		Calculates adjusted r squared value for fit_results
		'''
		ss_tot = sum((self.data['y']-np.mean(self.data['y']))**2)
		ss_res = sum(self.fit_results.fun**2)
		# n is the number of points
		n = len(self.data['y'])
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
		self.peak_x_pos = \
			self.data['x'].to_numpy()[np.argmax(self.data['y_hat'])]
		self.y_peak_height = np.max(self.data['y_hat'])

	def _calc_typical_threshold(self, gaussian_number):
		'''
		Calculates the most commonly used threshold,
		mean + sd * self.sd_multiplier, 
		of one of the best-fit gaussians (identified by gaussian_number)
		'''
		mean_param = 'mu_' + str(gaussian_number)
		sd_param = 'sigma_' + str(gaussian_number)
		threshold = \
			self.fit_result_dict[mean_param] + \
			self.sd_multiplier*self.fit_result_dict[sd_param]
		return(threshold)

	def _perform_fit(self):
		'''
		Performs fit with mixture of two gaussians and runs calculation
		of adjusted r squared and peak of distribution mixture
		'''
		# no unittest needed here
		self._id_starting_vals()
		self.fit_density()
		self._calc_fit_adj_rsq()
		self._find_peak()

	def _calc_mu_distance_to_peak(self):
		'''
		Calculates vector of distances of [mu_1, mu_2] to the peak x
		position of the fit
		'''
		mu_to_peak_distvec = np.abs(
			self.peak_x_pos - \
			np.array(
				[self.fit_result_dict['mu_'+str(dist_count)] for dist_count in self.dist_num_list]
				)
			)
		return(mu_to_peak_distvec)

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
		threshold_vec = np.array(
			[self._calc_typical_threshold(dist_count) for dist_count in self.dist_num_list]
			)
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
			sigma_vec = np.array(
				[self.fit_result_dict['sigma_'+str(dist_count)] for dist_count in self.dist_num_list]
				)
			threshold = threshold_vec[np.argmin(sigma_vec)]
		return(threshold)

	def _get_indiv_fit_df_for_plot(self, dist_count):
		'''
		Return df with x and y values for distribution corresponding 
		to dist_count
		'''
		indiv_dist_df = pd.DataFrame({
			'x': self.data['x'],
			'y':
			self._single_gauss_calculator(
				self.data['x'],
				self.fit_result_dict['lambda_'+str(dist_count)],
				self.fit_result_dict['mu_'+str(dist_count)],
				self.fit_result_dict['sigma_'+str(dist_count)]
				),
			'data_type': 'individual fit Gaussians',
			'linetype': 'dashed',
			'id': 'gauss'+str(dist_count)
			})
		return(indiv_dist_df)

	def plot(self):
		'''
		Plot threshold identification graph
		'''
		original_df = pd.DataFrame({'x': self.data['x'], 'y': self.data['y'],
			'id': 'smoothed data', 'linetype': 'solid'})
		original_df['data_type'] = original_df['id']
		fitted_df = pd.DataFrame({'x': self.data['x'], 'y': self.data['y_hat'],
			'id': 'combined fit model', 'linetype': 'solid'})
		fitted_df['data_type'] = fitted_df['id']
		indiv_df_list = [
			self._get_indiv_fit_df_for_plot(dist_count) for 
			dist_count in self.dist_num_list
			]
		combined_df = \
			pd.concat([original_df, fitted_df] + indiv_df_list,
				sort = True)
		color_dict = \
			{'smoothed data': '#377eb8', 'combined fit model': '#e41a1c',
				'individual fit Gaussians': '#4daf4a'}
#		color_dict = ['#377eb8', '#e41a1c', '#4daf4a']
		p = self._create_ggplot(combined_df, color_dict)
		return(p)

class _TwoGaussianFitThresholdMethod(_GaussianFitThresholdMethod):
	'''
	Generic class for methods for finding a threshold that involve
	fitting a mixture of two gaussians
	'''

	def __init__(self, method_name, threshold_flag, x_vals, y_vals,
		lower_bounds, upper_bounds, sd_multiplier_val):
		self.param_idx_dict = \
			{'lambda_1': 0, 'mu_1': 1, 'sigma_1': 2, 'lambda_2': 3, 'mu_2': 4,
				'sigma_2': 5}
		dist_num = 2
		super(_TwoGaussianFitThresholdMethod, self).__init__(
			method_name, threshold_flag, x_vals, y_vals, lower_bounds, upper_bounds, dist_num, sd_multiplier_val)

	def _id_starting_vals(self):
		'''
		Identifies starting values for parameters

		Creates self.starting_param_vals, a numpy array of starting
		values with indices corresponding to those in
		self.param_idx_dict

		Totally heuristic; will not reproduce matlab's behavior
		'''
		self.starting_param_vals = np.zeros(len(self.param_idx_dict))
		# first mean is ~1/20 of the way to this x value
		self.starting_param_vals[self.param_idx_dict['mu_1']] = self.x_span/20
		# second mean is ~1/2 of the way to this x value
		self.starting_param_vals[self.param_idx_dict['mu_2']] = self.x_span/2
		# highest peak is highest value of y
		self.starting_param_vals[self.param_idx_dict['lambda_1']] = self.max_y
		# second highest peak is the value of y closest to mu_2
		mu_2_closest_idx = \
			(np.abs(self.data['x'] - self.starting_param_vals[
				self.param_idx_dict['mu_2']])).argmin()
		self.starting_param_vals[self.param_idx_dict['lambda_2']] = \
			self.data['y'][mu_2_closest_idx]
		self.starting_param_vals[self.param_idx_dict['sigma_1']] = \
			self.x_span/12
		self.starting_param_vals[self.param_idx_dict['sigma_2']] = \
			self.x_span/6
		# check that starting parameters between bounds
		self._check_starting_vals()

	def _digauss_calculator(self, x, lambda_1, mu_1, sigma_1, lambda_2, mu_2, sigma_2):
		'''
		Computes sum of two gaussians with weights lamdba_1 and
		lambda_2, respectively
		'''
		y = self._single_gauss_calculator(x, lambda_1, mu_1, sigma_1) + \
			self._single_gauss_calculator(x, lambda_2, mu_2, sigma_2)
		return(y)

	def _residual_fun(self, params):
		'''
		Calculates residuals of difference between
		_digauss_calculator(x) and y_data 
		'''
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			y_params = \
				self._digauss_calculator(self.data.x,
					params[self.param_idx_dict['lambda_1']],
					params[self.param_idx_dict['mu_1']],
					params[self.param_idx_dict['sigma_1']],
					params[self.param_idx_dict['lambda_2']],
					params[self.param_idx_dict['mu_2']],
					params[self.param_idx_dict['sigma_2']])
		residuals = self.data.y - y_params
		return(residuals)

class _ThreeGaussianFitThresholdMethod(_GaussianFitThresholdMethod):
	'''
	Generic class for methods for finding a threshold that involve
	fitting a mixture of two gaussians
	'''

	def __init__(self, method_name, threshold_flag, x_vals, y_vals,
		lower_bounds, upper_bounds, sd_multiplier_val):
		self.param_idx_dict = \
			{'lambda_1': 0, 'mu_1': 1, 'sigma_1': 2, 'lambda_2': 3, 'mu_2': 4,
				'sigma_2': 5, 'lambda_3': 6, 'mu_3': 7, 'sigma_3': 8}
		dist_num = 3
		super(_ThreeGaussianFitThresholdMethod, self).__init__(
			method_name, threshold_flag, x_vals, y_vals, lower_bounds, upper_bounds, dist_num, sd_multiplier_val)

	def _id_starting_vals(self):
		'''
		Identifies starting values for parameters

		Creates self.starting_param_vals, a numpy array of starting
		values with indices corresponding to those in
		self.param_idx_dict

		Totally heuristic; will not reproduce matlab's behavior
		'''
		self.starting_param_vals = np.zeros(len(self.param_idx_dict))
		# first mean is ~1/20 of the way to this x value
		self.starting_param_vals[self.param_idx_dict['mu_1']] = self.x_span/20
		# third mean is ~1/2 of the way to this x value
		self.starting_param_vals[self.param_idx_dict['mu_3']] = self.x_span/2
		# second mean is average of the two values
		self.starting_param_vals[self.param_idx_dict['mu_2']] = \
			np.mean([
				self.starting_param_vals[self.param_idx_dict['mu_1']],
				self.starting_param_vals[self.param_idx_dict['mu_3']]
				])
		# highest peak is highest value of y
		self.starting_param_vals[self.param_idx_dict['lambda_1']] = self.max_y
		# second and third highest peaks are the value of y closest to 
		# mu_2 and mu_3, respectively
		mu_2_closest_idx = \
			(np.abs(self.data['x'] - self.starting_param_vals[
				self.param_idx_dict['mu_2']])).argmin()
		mu_3_closest_idx = \
			(np.abs(self.data['x'] - self.starting_param_vals[
				self.param_idx_dict['mu_3']])).argmin()
		self.starting_param_vals[self.param_idx_dict['lambda_2']] = \
			self.data['y'][mu_2_closest_idx]
		self.starting_param_vals[self.param_idx_dict['lambda_3']] = \
			self.data['y'][mu_3_closest_idx]
		# total heuristic for sigma starting vals for dists 1-3
		self.starting_param_vals[self.param_idx_dict['sigma_1']] = \
			self.x_span/12
		self.starting_param_vals[self.param_idx_dict['sigma_3']] = \
			self.x_span/6
		self.starting_param_vals[self.param_idx_dict['sigma_2']] = \
			np.mean([
				self.starting_param_vals[self.param_idx_dict['sigma_1']],
				self.starting_param_vals[self.param_idx_dict['sigma_3']]
				])
		# check that starting parameters between bounds
		self._check_starting_vals()

	def _trigauss_calculator(
		self, x,
		lambda_1, mu_1, sigma_1,
		lambda_2, mu_2, sigma_2,
		lambda_3, mu_3, sigma_3):
		'''
		Computes sum of three gaussians with weights lamdba_1-lambada_3
		'''
		y = self._single_gauss_calculator(x, lambda_1, mu_1, sigma_1) + \
			self._single_gauss_calculator(x, lambda_2, mu_2, sigma_2) + \
			self._single_gauss_calculator(x, lambda_3, mu_3, sigma_3)
		return(y)

	def _residual_fun(self, params):
		'''
		Calculates residuals of difference between
		_trigauss_calculator(x) and y_data 
		'''
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			y_params = \
				self._trigauss_calculator(self.data.x,
					params[self.param_idx_dict['lambda_1']],
					params[self.param_idx_dict['mu_1']],
					params[self.param_idx_dict['sigma_1']],
					params[self.param_idx_dict['lambda_2']],
					params[self.param_idx_dict['mu_2']],
					params[self.param_idx_dict['sigma_2']],
					params[self.param_idx_dict['lambda_3']],
					params[self.param_idx_dict['mu_3']],
					params[self.param_idx_dict['sigma_3']])
		residuals = self.data.y - y_params
		return(residuals)
		
class _mu1PosThresholdMethodTwoGauss(_TwoGaussianFitThresholdMethod):
	### !!! NEEDS BETTER METHOD DESCRIPTION

	def __init__(self, x_vals, y_vals, sd_multiplier_val = sd_multiplier):
		method_name = \
			'mu_1+'+\
			str(sd_multiplier_val)+\
			'*sigma_1[mu_1-positive]_two-gaussian'
		lower_bounds = np.array(
			[1, 0, sys.float_info.min, 0.5, -np.inf, sys.float_info.min])
		upper_bounds = np.array([np.inf]*6)
		threshold_flag = 0
		super(_mu1PosThresholdMethodTwoGauss, self).__init__(
			method_name, threshold_flag, x_vals, y_vals, lower_bounds,
			upper_bounds, sd_multiplier_val)

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
			# should go ahead with b1+self.sd_multiplier*c1
			self.method_name = \
				'mu_1+'+\
				str(self.sd_multiplier)+\
				'*sigma_1[mu_1-positive]_poor_minor_fit_two-gaussian'
			self.threshold_flag = 5;
			self.threshold = self._calc_typical_threshold(1)
		else:
			self.threshold = np.nan

class _mu1PosThresholdMethodThreeGauss(_ThreeGaussianFitThresholdMethod):
	### !!! NEEDS BETTER METHOD DESCRIPTION

	def __init__(self, x_vals, y_vals, sd_multiplier_val = sd_multiplier):
		method_name = \
			'mu_1+'+\
			str(sd_multiplier_val)+\
			'*sigma_1[mu_1-positive]_three-gaussian'
		lower_bounds = np.array([
			1, 0, sys.float_info.min,
			0.5, 0, sys.float_info.min,
			0.5, -np.inf, sys.float_info.min
			])
		upper_bounds = np.array([np.inf]*9)
		threshold_flag = 0
		super(_mu1PosThresholdMethodThreeGauss, self).__init__(
			method_name, threshold_flag, x_vals, y_vals, lower_bounds,
			upper_bounds, sd_multiplier_val)

	def _id_threshold(self):
		'''
		Identify threshold
		'''
		mu_to_peak_distvec = self._calc_mu_distance_to_peak()
		if self.rsq_adj > self.good_fit_rsq:
			self.threshold = \
				self._find_threshold_with_distant_peaks(mu_to_peak_distvec)
		elif np.abs(self.fit_result_dict['lambda_1']-self.y_peak_height) <= \
			self._vertical_close_to_peak_dist and \
			mu_to_peak_distvec[0] < self._close_to_peak_dist:
			# poor r sq because the smaller peak fit poor
			# but as long as the major peak fit good, as est by the cond here
			# should go ahead with b1+self.sd_multiplier*c1
			self.method_name = \
				'mu_1+'+\
				str(self.sd_multiplier)+\
				'*sigma_1[mu_1-positive]_poor_minor_fit_three-gaussian'
			self.threshold_flag = 5;
			self.threshold = self._calc_typical_threshold(1)
		else:
			self.threshold = np.nan

class _mu1ReleasedThresholdMethod(_TwoGaussianFitThresholdMethod):
	### !!! NEEDS BETTER METHOD DESCRIPTION

	def __init__(self, x_vals, y_vals, sd_multiplier_val = sd_multiplier):
		method_name = 'mu_1+'+str(sd_multiplier_val)+'*sigma_1[mu_1-released]'
		lower_bounds = np.array(
			[1, -np.inf, sys.float_info.min, 0.5, -np.inf, sys.float_info.min])
		upper_bounds = np.array([np.inf]*6)
		threshold_flag = 2
		super(_mu1ReleasedThresholdMethod, self).__init__(
			method_name, threshold_flag, x_vals, y_vals, lower_bounds,
			upper_bounds, sd_multiplier_val)

	def _id_threshold(self):
		'''
		Identify threshold
		'''
		mu_to_peak_distvec = self._calc_mu_distance_to_peak()
		if self.rsq_adj > self.good_fit_rsq and \
			(self.fit_result_dict['mu_1'] > 0 or \
				self.fit_result_dict['mu_2'] > 0):
			# if fit is good and both means positive, calculate
			# threshold based on b1+self.sd_multiplier*c1 same way as in
			# _mu1PosThresholdMethodTwoGauss
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
				# vals of the image), causing problems using
				# mu+self.sd_multiplier*sigma
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
		# TODO: test lowering the lower bound to the same calculation
		# as is used in gaussian methods for minimum peak x position
		y_data_above_0 = np.cumprod(self.data['y'].to_numpy()>0).astype(bool)
		last_x_above_0 = np.max(self.data['x'].to_numpy()[y_data_above_0])
		self._lower_bound = 0.05 * last_x_above_0		
		self._upper_bound = 0.8 * last_x_above_0
		# only include every xstep-th value in sliding circle
		# the idea here is to save time by not including nearly
		# identical values
		self._xstep = xstep
			# TODO: It might be good to set this based on autocorr
			# in the future, i.e. find distance at which
			# autocorrelation of ln histogram falls off by a certain
			# amount and use that
		# factors by which to stretch x and y dimensions
		self._x_stretch_factor = 0.1
		self._y_stretch_factor = 100
		# number of neighboring circles to sum/average over when
		# calculating threshold position
		self._area_sum_sliding_window_size = 5
		# proportion of x-axis points to include in radius
		self._radial_prop = 0.15

	def _find_xstep(self, element_num, xstep_multiplier):
		'''
		Finds xstep given xstep_multiplier
		Minimum possible value returned is 1
		'''
		xstep = np.max([1, np.floor(element_num * xstep_multiplier)]
			).astype(int)
		return(xstep)

	def _sample_and_stretch_graph(self):
		'''
		Subsamples and stretches graph to be used for sliding circle
		'''
		# take every x_step-th value of xData and yData, multiply by
		# respective stretch factors
		self._x_vals_stretched = \
			self.data['x'].to_numpy()[0::self._xstep] * self._x_stretch_factor
		self._y_vals_stretched = \
			self.data['y'].to_numpy()[0::self._xstep] * self._y_stretch_factor
		# calculate ceiling on max values of stretched x and y
		self._x_stretched_max_int = int(np.ceil(np.max(self._x_vals_stretched)))
		self._y_stretched_max_int = int(np.ceil(np.max(self._y_vals_stretched)))

	def _create_poly_mask(self):
		'''
		Creates a mask where area under (stretched) curve is white, area
		above curve is black
		'''
		# pad ends of stretched, subsampled x and y vals so that they
		# loop back on themselves (to allow polygon creation)
		x_poly = np.concatenate([[0], self._x_vals_stretched,
			[self._x_vals_stretched[-1]], [0]])
		y_poly = np.concatenate([[0], self._y_vals_stretched, [0, 0]])
		# create mask in which area under curve is white, area over
		# curve black, but upside down
		# (python code based on https://stackoverflow.com/a/3732128,
		# modified to reproduce matlab behavior)
		poly_img = Image.new('L',
			(self._x_stretched_max_int, self._y_stretched_max_int), 0)
			# create blank image of all 0s
		polygon = list(zip(x_poly, y_poly))
			# create list of tuples of every x and y value
		ImageDraw.Draw(poly_img).polygon(polygon, outline=1, fill=1)
			# draw polygon based on x-y coordinates
		# NB on ImageDraw.Draw.polygon behavior: if each position in the
		# output matrix is a grid square, ImageDraw treats (0,0) as the
		# upper left corner of the (0,0) square. All positions at
		# integers values are drawn within the top and left corners of
		# the corresponding coordinate box, with floats being rounded to
		# the nearest ~10^-15 (see unittest for more info)
		_fit_im_unpadded = np.array(poly_img, dtype = bool)
		# pad self._fit_im with circle_radius on each side
		col_num = _fit_im_unpadded.shape[1]
		row_num = _fit_im_unpadded.shape[0]
		fit_im_row_indices = np.concatenate((
			[0]*self._radius,
			np.arange(0,row_num)
			))
		fit_im_col_indices = np.concatenate((
			[0]*self._radius,
			np.arange(0,col_num)
			))
		# pad left and top (i.e. left bottom of log hist plot) of array
		# with self._radius number of copies of left-most and top-most
		# row, respectively 
		fit_im_row_padded = _fit_im_unpadded[fit_im_row_indices,:]
		fit_im_col_padded = fit_im_row_padded[:, fit_im_col_indices]
		# pad right and bottom of array (i.e. right and top of log hist
		# plot) with zeros
		self._fit_im = np.pad(
			fit_im_col_padded,
			((0,self._radius),(0,self._radius)),
			'constant'
			)

	def _create_circle_mask(self, center_x, center_y, radius, im_width,
		im_height):
		'''
		Creates a white mask on a black im_width x im_height background
		that is a circle centered on (center_x, center_y) of specified
		radius
		Returns mask as np array
		'''
		# create matrices of distances of every point from center along
		# x and y axes
		# shift center by 0.5 in both directions to center circle in the
		# point in gridspace where the corresponding polygon coordinate
		# would be in self._fit_im
		# Corresponding matlab PIE code behaves identically, but without
		# the shift in center values, and without padding in self._fit_im
		circle_mask = np.zeros((im_height+2*radius, im_width+2*radius), dtype = bool)
		# only calculate mask within a reasonable window around the
		# center, or it will be too computationally expensive/slow
		# create arrays of distances from a shifted center point,
		x_cent_dist_list_full = np.arange(0, im_width + radius*2) - (center_x - 0.5) - radius
		y_cent_dist_list_full = np.arange(0, im_height + radius*2) - (center_y - 0.5) - radius
#		x_cent_dist_list = \
#			x_cent_dist_list_full[
#				np.logical_and(
#					x_cent_dist_list_full >= -radius,
#					x_cent_dist_list_full <= radius
#					)
#				]
#		y_cent_dist_list = \
#			y_cent_dist_list_full[
#				np.logical_and(
#					y_cent_dist_list_full >= -radius,
#					y_cent_dist_list_full <= radius
#					)
#				]
#		# tile arrays of distances from
#		x_center_dist_mat = np.tile(x_cent_dist_list, [len(y_cent_dist_list), 1])
#		y_center_dist_mat = \
#			np.tile(np.reshape(y_cent_dist_list, [len(y_cent_dist_list), 1]),
#				[1, len(x_cent_dist_list)])
#		# identify points whose distance from center is less than or
#		# equal to radius
#		circle_minimask = \
#			(np.square(x_center_dist_mat) + np.square(y_center_dist_mat)) \
#				<= radius**2
#		# place circle_minimask in correct position relative to
#		# self._fit_im
#		x_mask_bottom_idx = np.floor(center_x + x_cent_dist_list[0]).astype(int)
#		x_mask_top_idx = np.ceil(center_x + x_cent_dist_list[-1]).astype(int)
#		y_mask_bottom_idx = np.floor(center_y + y_cent_dist_list[0]).astype(int)
#		y_mask_top_idx = np.ceil(center_y + y_cent_dist_list[-1]).astype(int)
#		circle_mask[y_mask_bottom_idx:y_mask_top_idx,
#			x_mask_bottom_idx: x_mask_top_idx] = circle_minimask
		x_center_dist_mat = np.tile(x_cent_dist_list_full, [len(y_cent_dist_list_full), 1])
		y_center_dist_mat = \
			np.tile(np.reshape(y_cent_dist_list_full, [len(y_cent_dist_list_full), 1]),
				[1, len(x_cent_dist_list_full)])
		circle_mask = (np.square(x_center_dist_mat) + np.square(y_center_dist_mat)) \
				<= radius**2
		return(circle_mask)

	def _id_circle_centers(self):
		'''
		Identifies x positions between lower and upper bound, and
		corresponding y positions, as centers of circles to overlap
		with self._fit_im
		'''
		x_center_bool = np.logical_and(
			(self._x_vals_stretched >
				self._lower_bound * self._x_stretch_factor),
			(self._x_vals_stretched <
				self._upper_bound * self._x_stretch_factor))
		self._x_centers = self._x_vals_stretched[x_center_bool]
		self._y_centers = self._y_vals_stretched[x_center_bool]
		# specify radius, in number of pixels
		# (i.e. number of points along x-axis scaled by distance between points)
		self._radius = np.max([2,np.round(np.count_nonzero(x_center_bool)*self._radial_prop*np.mean(np.diff(self._x_centers))).astype(int)])

	def _calculate_circle_areas(self):
		'''
		Slides circle along ridge of self._fit_im and calculates area
		under the curve within each circle
		'''
		# keep track of area inside circle at each position along
		# x_centers
		self._inside_area = np.zeros(len(self._x_centers))
		# loop through center positions, identify circle mask, and
		# calculate area of self._fit_im within that circle
		for idx, (cx, cy) in enumerate(zip(self._x_centers, self._y_centers)):
			current_circle_mask = \
				self._create_circle_mask(cx, cy, self._radius,
					self._x_stretched_max_int, self._y_stretched_max_int)
			mask_overlap = np.logical_and(current_circle_mask, self._fit_im)
			self._inside_area[idx] = \
				np.count_nonzero(mask_overlap)/ \
				np.count_nonzero(current_circle_mask)
#			if idx == 0:
#				msk = np.logical_xor(current_circle_mask, self._fit_im)
#				cv2.imwrite('/Users/plavskin/Documents/circle_0.tif', 255*np.uint8(msk))
#			if self._inside_area[idx] == np.max(self._inside_area):
#				msk = np.logical_xor(current_circle_mask, self._fit_im)
#				cv2.imwrite('/Users/plavskin/Documents/circle_max.tif', 255*np.uint8(msk))
#		msk = np.logical_xor(current_circle_mask, self._fit_im)
#		cv2.imwrite('/Users/plavskin/Documents/circle_last.tif', 255*np.uint8(msk))

	def _perform_fit(self):
		'''
		Performs circle sliding procedure
		'''
		self._sample_and_stretch_graph()
		self._id_circle_centers()
		self._create_poly_mask()
		self._calculate_circle_areas()

	def _id_threshold(self):
		'''
		Identify threshold
		'''
		# sum over each self._area_sum_sliding_window_size
		# neighboring points
		if len(self._inside_area) <= self._area_sum_sliding_window_size:
			convolution_window = np.ones(len(self._inside_area))
		else:
			convolution_window = np.ones(self._area_sum_sliding_window_size)
		sum_inside_area = \
			np.convolve(self._inside_area, convolution_window, mode = 'same')
		# threshold is the x position at which sum_inside_area is
		# highest, corrected for the 'stretch factor' originally used
		self.threshold = \
			self._x_centers[np.argmax(sum_inside_area)] / self._x_stretch_factor

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
		xstep = self._find_xstep(len(x_vals), 0.001)
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

	def plot(self):
		'''
		Plot threshold identification graph
		'''
		original_df = pd.DataFrame({
			'x': self.data['x'],
			'y': self.data['y'],
			'id': 'smoothed data', 'linetype': 'solid'})
		original_df['data_type'] = original_df['id']
		color_dict = \
			{'smoothed data': '#377eb8'}
		p = self._create_ggplot(original_df, color_dict)
		return(p)

class _FitSlidingCircleThresholdMethod(_SlidingCircleThresholdMethod):
	'''
	Threshold method that takes in best fit to histogram of tophat
	image and finds threshold using sliding circle method
	If no best fit is provided, performs its own fit via
	_mu1ReleasedThresholdMethod
	'''
	def __init__(self, x_vals, y_original, y_model = None):
		'''
		y_original is the smoothed log histogram
		y_model is the result of the best fit of a mixture of two
		gaussians to the smoothed log histogram
		'''
		threshold_flag = 3
		method_name = 'sliding_circle_data'
		#xstep = self._find_xstep(len(x_vals), 0.1)
		xstep = self._find_xstep(len(x_vals), 0.03)
			# heuristic - see TODO in parent class
			# NB: current implementation will not exactly reproduce
				# matlab code, which hard-codes the xstep as either 3 or
				# 100, for data vs fit-based sliding circle, respectively
				# In fact, relative xstep for this method has been
				# decreased 3x
		if y_model is None:
			y_model = self._fit_mu1Released(x_vals, y_original)
		self.y_original = y_original
		super(_FitSlidingCircleThresholdMethod, self).__init__(
			method_name, threshold_flag, x_vals, y_model, xstep)

	def _fit_mu1Released(self, x, y):
		'''
		Performs fit with _mu1ReleasedThresholdMethod, returns fitted y
		values
		'''
		### !!! NEEDS UNITTEST
		threshold_method = _mu1ReleasedThresholdMethod(x, y)
		threshold_method._perform_fit()
		y_model = threshold_method.data.y_hat
		return(y_model)

	def plot(self):
		'''
		Plot threshold identification graph
		'''
		original_df = pd.DataFrame({'x': self.data['x'], 'y': self.y_original,
			'id': 'smoothed data', 'linetype': 'solid'})
		original_df['data_type'] = original_df['id']
		fitted_df = pd.DataFrame({'x': self.data['x'], 'y': self.data['y'],
			'id': 'combined fit model', 'linetype': 'solid'})
		fitted_df['data_type'] = fitted_df['id']
		combined_df = \
			pd.concat([original_df, fitted_df], sort = True)
		color_dict = \
			{'smoothed data': '#377eb8', 'combined fit model': '#e41a1c'}
		p = self._create_ggplot(combined_df, color_dict)
		return(p)

def threshold_image(input_im, image_type, cell_intensity_num, return_plot = False):
	'''
	Reads in input_im and returns an automatically thresholded bool mask
	If return_plot is true, also returns a plotnine plot object
	'''
	### !!! NEEDS UNITTEST
	threshold_finder = _ThresholdFinder(input_im, image_type, cell_intensity_num)
	try:
		threshold_mask = threshold_finder.get_threshold_mask()
		threshold_method_name = threshold_finder.threshold_method.method_name
		threshold = threshold_finder.threshold_method.threshold
		# check whether default threshold method was used
		threshold_class_string = \
			threshold_finder.threshold_method.__class__.__name__
		default_threshold_method_usage = \
			threshold_class_string == \
			threshold_finder.default_threshold_method_class
		if return_plot or not default_threshold_method_usage:
			threshold_plot = threshold_finder.threshold_method.plot()
		else:
			threshold_plot = None
	except ValueError as e:
#		if str(e) == '3 or fewer unique values in tophat image':
		# return empty mask
		threshold_mask = np.zeros(np.shape(input_im), dtype = bool)
		threshold_method_name = 'Error: ' + str(e)
		threshold_plot = None
		threshold = 0
		default_threshold_method_usage = False
	return(threshold_mask, threshold_method_name, threshold_plot,
		threshold, default_threshold_method_usage)

### !!! TODO: GET RID OF FLAGS


if __name__ == '__main__':

	pass
	# TODO: need to read in image file name via argparse, load image, pass it to threshold_finder, get back threshold mask

