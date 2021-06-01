#!/usr/bin/python

'''
Performs thresholding to split colonies into fluorescent vs
non-fluorescent classes
'''

import numpy as np
import pandas as pd
import plotnine as p9
import scipy.stats as stats
import warnings
from PIE.density_fit import DensityFitterMLE
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

class _FluorDensityFitter(DensityFitterMLE):
	'''
	Parent class for classes that find mixture of two distributions
	that best fit provided input_vector
	'''
	def __init__(self, input_vector):
		self.raw_data = input_vector
		# check if there's sufficient data to perform fit
		if len(np.unique(self.raw_data)) > 1:
			self.sufficient_data = True
		else:
			self.sufficient_data = False
		# set linear and nonlinear constraints to None
		self.linear_constraints = None
		self.nonlinear_constraints = None
		# set top quantile to include in fitting data (heuristic)
		self.top_quantile=1-10**-4
		# set density kernel bandwidth relative to data range
		# to try as start
		self._rel_start_bandwidth=0.001
		# set relative starting step size
		self._rel_start_step_size=0.25
		# set number of gridpoints to use for KDE
		# (KDE probably faster if power of 2)
		self._grid_points=1024

	def _prep_fit(self):
		'''Prepares data, bounds, and starting points for fit'''
		# trim data 
		self._prep_data()
		# calculate gaussian kernel of data
		self._id_two_peak_kernel()
		if self.sufficient_data:
			# find properties of kernel that will help ID parameter
			# starting points
			self._calculate_prestart_positions()
			# id bounds and starting parameters for fit
			self._id_starting_vals()
			self._id_bounds()
			self._check_starting_vals()

	def _prep_data(self):
		'''
		Trim highest-fluor data before fitting curves to remove values
		corresponding to dead cells and random bright pixels
		'''
		if self.sufficient_data:
			top_quantile_val=np.nanquantile(self.raw_data,self.top_quantile)
			non_na_data = self.raw_data[~np.isnan(self.raw_data)]
			trimmed_data=non_na_data[non_na_data < top_quantile_val]
		else:
			trimmed_data=self.raw_data
		self.data=trimmed_data

	def _peak_finder(self, y):
		'''
		Counts number of local peaks in a vector of y-values
		'''
		if len(y.shape) > 1:
			raise ValueError('y must be a 1D vector')
		# find peaks that are not at edges of the vector
		d_y=np.diff(y)
		sign=np.sign(d_y)
		middle_peak_bool=np.diff(sign) == -2
		peak_idx=np.where(middle_peak_bool)[0]
		# if first position is a peak, add it to peaks
		if y[0] > y[1]:
			peak_idx=np.append(0, peak_idx)
		# if last position is a peak, add it to peaks
		if y[-1] > y[-2]:
			peak_idx=np.append(peak_idx, (len(y)-1))
		return(peak_idx)

	def _calculate_kernel(self, x_grid, bw):
		'''
		Calculates gaussian kernel density around data
		'''
		data_no_nan = self.data[~np.isnan(self.data)]
		# bandwidth is multiplied by std of data within gaussian_kde
		bw_mod = bw/data_no_nan.std()
		# instantiate and fit the KDE model
		kde=gaussian_kde(data_no_nan, bw_method=bw_mod)
		pdf = kde.pdf(x_grid)
		return(pdf)

	def _id_two_peak_kernel(self):
		'''
		Finds kernel with smallest bandwidth that gives a density with
		2 peaks
		'''
		data_min=np.min(self.data)
		data_max=np.max(self.data)
		data_extent=data_max - data_min
		# calculate x grid over which to apply kernel
		# add space on either side in case peaks are at extremes
		x_grid=np.linspace(
			data_min-data_extent/3,
			data_max+data_extent/3,
			self._grid_points
			)
		# start at a low bandwidth and increase until hitting 2 peaks
		# if hit 3 peaks first, decrease step and decease bw, etc
		# after changing directions like this 10 times, give up and use
		# distribution with 3 peaks
		# calculate starting bandwidth to use for kernel
		bw=self._rel_start_bandwidth*data_max
		if self.sufficient_data:
			# initialize other parameters
			rel_step_size=self._rel_start_step_size
			search_complete=False
			direction_flips=0
			max_direction_flips=10
			while not search_complete:
				density=self._calculate_kernel(x_grid, bw)
				peak_idx=self._peak_finder(density)
				peak_num=len(peak_idx)
				if peak_num==2:
					search_complete=True
				elif peak_num > 2:
					if peak_num == 3 and direction_flips >= max_direction_flips:
						# if we've been searching for a while, maybe 
						# it's not possible to get exactly 2 peaks in
						# this distribution
						search_complete=True
					else:
						# increase bandwidth
						bw=bw*(1+rel_step_size)
				else:
					# only one peak, need to decrease bandwidth again,
					# but by less
					rel_step_size=rel_step_size*0.9
					direction_flips=direction_flips+1
					bw=bw*(1-rel_step_size)
			# only include first and last peaks (in case there are more
			# than 2)
			self.peak_idx=peak_idx[[0,-1]]
		else:
			density = self._calculate_kernel(x_grid, bw)
			self.peak_idx=np.array([np.nan, np.nan])
#		self._bandwidth=bw
		self._kernel_df=pd.DataFrame({
			'fluor': x_grid,
			'density': density
			})
		
	def _inflection_dist_finder(self, peak_idx, side):
		'''
		Identifies distance between peak and rising inflection point
		(max of gradient) of density on one side of x[peak_idx]

		side can be 'left' or 'right'
		'''
		x=self._kernel_df.fluor.to_numpy()
		density=self._kernel_df.density.to_numpy()
		if side == 'left':
			mask_bool=np.arange(x.size)>peak_idx
		elif side == 'right':
			mask_bool=np.arange(x.size)<peak_idx
		else:
			raise ValueError('side must be left or right')
		# calculate gradient of density
		grad=np.diff(density, append=0)
		# create masked gradient to only look at values on one side of
		# peak
		masked_grad=np.ma.masked_array(grad, mask=mask_bool)
		# find index of maximum of masked gradient
		if side == 'left':
			grad_max_idx=np.argmax(masked_grad)
		elif side == 'right':
			grad_max_idx=np.argmin(masked_grad)
		# calculate inflection pt as the midpoint of the two points
		# with the max grad diff between them
		inflection_pt=np.mean(x[[grad_max_idx, grad_max_idx+1]])
		inflection_dist=np.abs(x[peak_idx] - inflection_pt)
		return(inflection_dist)

	def _calculate_prestart_positions(self):
		'''
		Calculates positions of modes and inflection points of the two
		outermost kernel peaks
		'''
		self._prestart_df=pd.DataFrame({
			'peak_idx': self.peak_idx,
			'mode': self._kernel_df.fluor.to_numpy()[self.peak_idx],
			'inflection_dist': np.nan
			}, index = [0,1])
		# find good starting point to estimate dispersion-related
		# params of distribution by finding distance between mode and
		# its nearest inflection point
		# need to use inflection point on the far side of
		# distribution's mode relative to other distribution's mode, to
		# avoid as much as possible biases introduced by the
		# distributions mixing
		self._prestart_df.at[0, 'inflection_dist']=\
			self._inflection_dist_finder(
				self._prestart_df.at[0, 'peak_idx'],
				'left'
				)
		self._prestart_df.at[1, 'inflection_dist']=\
			self._inflection_dist_finder(
				self._prestart_df.at[1, 'peak_idx'],
				'right'
				)

	def _id_midpoint(self):
		'''
		Identifies fluorescence value of lower point between density
		peaks
		'''
		x=self._kernel_df.fluor.to_numpy()
		density=self._kernel_df.density.to_numpy()
		mask_bool=\
			(x < self._prestart_df.at[0, 'mode']) | \
			(x > self._prestart_df.at[1, 'mode'])
		masked_density=np.ma.masked_array(density, mask=mask_bool)
		midpoint_idx=np.argmin(masked_density)
		self._midpoint=x[midpoint_idx]

	def _id_bounds(self):
		'''
		Identify lower and upper bounds for fitting
		'''
		pass

	def _id_starting_vals(self):
		'''
		Identifies starting values for parameters

		Creates self.starting_param_vals, a numpy array of starting
		values with indices corresponding to those in
		self.param_idx_dict
		'''
		pass

	def _create_data_to_plot(self):
		'''
		Creates distribution data to plot
		'''
		pass

	def _get_pdf_vals(self, fl_vals, cell_type):
		'''
		Returns pdf value of distribution for cell_type at fl_val

		cell_type can be either 'back' or 'fl'

		fl_vals can be either a single numeric or a numpy array
		'''
		pass

	def _get_cdf_vals(self, fl_vals, cell_type, direction):
		'''
		Returns cdf value of distribution for cell_type either above or
		below fl_val, depending on direction

		cell_type can be either 'back' or 'fl'

		fl_vals can be either a single numeric or a numpy array

		direction can be 'above' or 'below'
		'''
		pass

	def _create_data_to_plot(self):
		'''
		Creates distribution data to plot
		'''
		plot_data_wide=self._kernel_df.drop(columns=['density'])
		plot_data_wide['background']=\
			self._get_pdf_vals(self._kernel_df['fluor'], 'back')
		plot_data_wide['fluorescent']=\
			self._get_pdf_vals(self._kernel_df['fluor'], 'fl')
		return(plot_data_wide)

	def plot(self, x_label, plot_write_path, width_in, height_in):
		'''
		Creates plot of estimated background and fluorescent densities 
		overlaid on histogram of data, with lines denoting thresholds
		for background vs fluorescent data
		'''
		plot_data_wide=self._create_data_to_plot()
		# tidy plot data
		plot_data=pd.melt(
			plot_data_wide,
			id_vars=['fluor'],
			value_vars=['background', 'fluorescent'],
			var_name='distribution_id',
			value_name='density'
			)
		# get histogram binwidth
		hist_bin_centers = np.histogram_bin_edges(self.data, bins='auto')
		bin_num = len(hist_bin_centers)
		density_plot = \
			p9.ggplot() + \
			p9.geom_histogram(
				data=pd.DataFrame({'fluor':self.data}),
				mapping=p9.aes(x='fluor',y='stat(density)'),
				color='gray',
				binwidth=bin_num,
				linetype='solid',
				alpha=0,
				size=.25
				) + \
			p9.geom_line(
				data=plot_data,
				mapping=p9.aes(
					x='fluor',
					y='density',
					color='distribution_id'
					)
				) + \
			p9.geom_vline(
				xintercept=self.background_threshold,
				color='black',
				linetype='dashed'
				) + \
			p9.geom_vline(
				xintercept=self.fluor_threshold,
				color='#11ba33',
				linetype='dashed'
				) + \
			p9.scale_x_continuous(name = x_label) + \
			p9.scale_color_manual(
				values = {'fluorescent':'#11ba33', 'background':'black'}
				) + \
			p9.theme_bw() + \
			p9.theme(
				legend_position=(0.75,0.7),
				legend_title=p9.element_blank()
				)
		density_plot.save(
			filename=plot_write_path,
			width=width_in,
			height=height_in,
			units='in',
			dpi=300)

	def _misid_ratio_back(self, fl_val):
		'''
		Calculates the proportion of cells from the background 
		distribution that would be misidentified as fluorescent at 
		fl_val

		Returns the difference between that proportion and 
		self.acceptable_proportion_misID
		'''
		prop_labeled_as_back=self._get_cdf_vals(fl_val, 'back', 'below')
		prop_labeled_as_fl=self._get_cdf_vals(fl_val, 'fl', 'below')
		misid_ratio=prop_labeled_as_fl/prop_labeled_as_back
		return(misid_ratio)
	
	def _misid_ratio_fl(self, fl_val):
		'''
		Calculates the proportion of cells from the fluorescent 
		distribution that would be misidentified as background at 
		fl_val
		'''
		prop_labeled_as_back=self._get_cdf_vals(fl_val, 'back', 'above')
		prop_labeled_as_fl=self._get_cdf_vals(fl_val, 'fl', 'above')
		misid_ratio=prop_labeled_as_back/prop_labeled_as_fl
		return(misid_ratio)

	def _dist_to_acceptable_prop_misID(self, fl_val, cell_type):
		'''
		Returns the difference between the proportion cells 
		misidentified as cell_type using a threshold of fl_val, and 
		self.acceptable_proportion_misID

		cell_type can be 'back' or 'fl'
		'''
		if cell_type=='back':
			misid_ratio = self._misid_ratio_back(fl_val)
		elif cell_type=='fl':
			misid_ratio = self._misid_ratio_fl(fl_val)
		else:
			raise ValueError('cell_type must be "back" or "fl"')
		prop_dist = np.abs(misid_ratio-self.acceptable_proportion_misID)
		return(prop_dist)

	def id_thresholds(self, threshold_output_file,
		acceptable_proportion_misID = 0.001):
		'''
		Based on parameters of distribution mix and 
		acceptable_proportion_misID, find cutoffs for the highest value 
		to be considered background and the lowest value to be 
		considered fluorescence, such that the proportion of cells 
		misidentified in either direction is as close as possible to 
		acceptable_proportion_misID

		This may miss some solution that allows for slightly smaller 
		proportion of misidentifications than 
		acceptable_proportion_misID, but avoids complicated tradeoffs 
		between the number of cells included and their purity
		'''
		self.acceptable_proportion_misID = acceptable_proportion_misID
		if self.sufficient_data:
			# fit density
			self.fit_density()
			# Find fluorescence value at which the proportion of cells from
			# the fluorescent distribution is closest possible to
			# allowed_misID_prop*(number of cells from background
			# distribution), and vice versa
			# use the 'midpoint' fluorescent value (lowest fluorescence
			# between two distributions' modes) as a starting point
			back_fit_result=minimize(
				self._dist_to_acceptable_prop_misID,
				self._midpoint,
				args = ('back')
				)
			fl_fit_result=minimize(
				self._dist_to_acceptable_prop_misID,
				self._midpoint,
				args = ('fl')
				)
			# safety mechanism in case real distribution is wider than
			# specified: set any overly optimistic threshold to the 
			# midpoint
			self.background_threshold=\
				np.min([back_fit_result.x[0], self._midpoint])
			self.fluor_threshold=\
				np.max([fl_fit_result.x, self._midpoint])
		else:
			self.fit_result_dict = dict(zip(
				self.param_idx_dict.keys(),
				[np.nan]*len(self.param_idx_dict.keys())
				))
			self.background_threshold = np.nan
			self.fluor_threshold = np.nan
		background_prop_misID=\
			self._misid_ratio_back(self.background_threshold)
		fluor_prop_misID=\
			self._misid_ratio_fl(self.fluor_threshold)
		# create bools of background and fluorescent colony IDs
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			self.background_bool = self.raw_data<self.background_threshold
			self.fluor_bool = self.raw_data>self.fluor_threshold
		background_colonies = float(np.sum(self.background_bool))
		fluor_colonies = float(np.sum(self.fluor_bool))
		total_colonies = float(len(self.raw_data))
		# write output
		output_dict={
			'proportion_background':background_colonies/total_colonies,
			'proportion_fluorescent':fluor_colonies/total_colonies,
			'proportion_unidentified':
				1-(background_colonies+fluor_colonies)/total_colonies,
			'background_threshold':self.background_threshold,
			'fluor_threshold':self.fluor_threshold,
			'background_prop_misID':background_prop_misID,
			'fluor_prop_misID':fluor_prop_misID
			}
		output_dict.update(self.fit_result_dict)
		output_df=pd.DataFrame(output_dict, index = [0])
		output_df.to_csv(threshold_output_file, index = False)

	def _perform_classification(self):
		'''
		Classifies self.raw_data
		'''
		self.classification_bool = \
			np.array([np.nan]*len(self.raw_data), dtype = object)
		self.classification_bool[self.raw_data < self.background_threshold] = \
			False
		self.classification_bool[self.raw_data > self.fluor_threshold] = True

	def classify_data(self, threshold_path, channel_name, plot_path,
		plot_width, plot_height):
		'''
		Returns a boolean numpy array, with every point in 
		self.raw_data that below the background threshold labeled 
		False, every point above the fluorescent threshold labeled 
		True, and every point in between labeled np.nan
		'''
		self._prep_fit()
		self.id_thresholds(threshold_path)
		self.plot(channel_name, plot_path, plot_width, plot_height)
		self._perform_classification()
		return(self.classification_bool)

class FluorLogisticFitter(_FluorDensityFitter):
	'''
	Fits mixture of two logistic distributions to data
	'''
	def __init__(self, input_vector):
		# set up parameter index dictionary
		# note that lambda_fl is just 1-lambda_back
		self.param_idx_dict={
			'lambda_back': 0,
			'location_back': 1,
			'scale_back': 2,
			'location_fl': 3,
			'scale_fl': 4}
		# the following parameters cannot have values below 0
		self.non_neg_params=\
			['location_back', 'location_fl', 'lambda_back']
		# the following parameters must be above 0
		self.above_zero_params=['scale_back', 'scale_fl']
		super(FluorLogisticFitter, self).__init__(input_vector)

	def _id_bounds(self):
		'''
		Identify lower and upper bounds for fitting
		'''
		self._id_midpoint()
		lower_bound_candidates=np.array([
			0,
			np.quantile(self.data,0.01),
			self.starting_param_vals[self.param_idx_dict['scale_back']]/10**2,
			self._midpoint,
			self.starting_param_vals[self.param_idx_dict['scale_fl']]/10**2
			])
		if self._midpoint < np.quantile(self.data,0.99):
			top_dist_val = np.quantile(self.data,0.99)
		else:
			top_dist_val = np.nanmax(self.data)
		upper_bound_candidates=np.array([
			1,
			self._midpoint,
			self.starting_param_vals[self.param_idx_dict['scale_back']]*10,
			top_dist_val,
			self.starting_param_vals[self.param_idx_dict['scale_fl']]*10
			])
		self.lower_bounds=self._check_bounds(lower_bound_candidates)
		self.upper_bounds=self._check_bounds(upper_bound_candidates)
		if not np.all(self.lower_bounds < self.upper_bounds):
			raise ValueError('Lower bounds must be below upper bounds')

	def _id_starting_vals(self):
		'''
		Identifies starting values for parameters

		Creates self.starting_param_vals, a numpy array of starting
		values with indices corresponding to those in
		self.param_idx_dict
		'''
		self.starting_param_vals=np.zeros(len(self.param_idx_dict))
		# location of logistic is its mode
		self.starting_param_vals[self.param_idx_dict['location_back']]=\
			self._prestart_df.at[0, 'mode']
		self.starting_param_vals[self.param_idx_dict['location_fl']]=\
			self._prestart_df.at[1, 'mode']
		# scale can be calculated as a function of distance between
		# mode and inflection point
		self.starting_param_vals[self.param_idx_dict['scale_back']]=\
			self._prestart_df.at[0, 'inflection_dist']/np.log(2+np.sqrt(3))
		self.starting_param_vals[self.param_idx_dict['scale_fl']]=\
			self._prestart_df.at[1, 'inflection_dist']/np.log(2+np.sqrt(3))
		# estimate lambda_fl first since peak of dist 2 less likely to
		# be affected by mixing from dist 1 (which should have smaller
		# scale)
		lambda_fl_start=\
			self._kernel_df.density[self._prestart_df.at[1, 'peak_idx']]*\
			4*self.starting_param_vals[self.param_idx_dict['scale_fl']]
		self.starting_param_vals[self.param_idx_dict['lambda_back']]=\
			1-lambda_fl_start

	def _dilogis_calculator(self, x, lambda_back, location_back, scale_back,
		location_fl, scale_fl):
		'''
		Computes likelihoods of x given the sum of two logistic
		distributions with weights lamdba_back and lambda_fl, respectively
		'''
		likelihoods=lambda_back*stats.logistic.pdf(x, location_back, scale_back) + \
			(1-lambda_back)*stats.logistic.pdf(x, location_fl, scale_fl)
		return(likelihoods)

	def _neg_ll_fun(self, params):
		'''
		Calculates negative log likelihood of observing raw_data given
		params
		'''
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
#			print(params)
			likelihoods=\
				self._dilogis_calculator(self.data,
					params[self.param_idx_dict['lambda_back']],
					params[self.param_idx_dict['location_back']],
					params[self.param_idx_dict['scale_back']],
					params[self.param_idx_dict['location_fl']],
					params[self.param_idx_dict['scale_fl']])
		neg_ll=-np.sum(np.log(likelihoods))
#		print(neg_ll)
		return(neg_ll)

	def _get_parameter_vals(self, cell_type):
		'''
		Returns dict of fitted parameters for cell_type, which can be 
		either 'back' or 'fl'
		'''
		param_dict = dict()
		if cell_type == 'back':
			param_dict['lambda']=self.fit_result_dict['lambda_back']
		elif cell_type == 'fl':
			param_dict['lambda']=1-self.fit_result_dict['lambda_back']
		else:
			raise ValueError('cell_type must be either "back" or "fl"')
		param_dict['location']=self.fit_result_dict['location_'+cell_type]
		param_dict['scale']=self.fit_result_dict['scale_'+cell_type]
		return(param_dict)

	def _get_pdf_vals(self, fl_vals, cell_type):
		'''
		Returns pdf value of distribution for cell_type at fl_val

		cell_type can be either 'back' or 'fl'

		fl_vals can be either a single numeric or a numpy array
		'''
		param_dict=self._get_parameter_vals(cell_type)
		pdf_vals=\
			param_dict['lambda']*\
			stats.logistic.pdf(
				fl_vals,
				param_dict['location'],
				param_dict['scale']
				)
		return(pdf_vals)

	def _get_cdf_vals(self, fl_vals, cell_type, direction):
		'''
		Returns cdf value of distribution for cell_type either above or
		below fl_val, depending on direction

		cell_type can be either 'back' or 'fl'

		fl_vals can be either a single numeric or a numpy array

		direction can be 'above' or 'below'
		'''
		param_dict=self._get_parameter_vals(cell_type)
		cdf_vals_below=\
			stats.logistic.cdf(
				fl_vals,
				param_dict['location'],
				param_dict['scale']
				)
		if direction == 'below':
			cdf_vals=param_dict['lambda']*cdf_vals_below
		elif direction == 'above':
			cdf_vals=param_dict['lambda']*(1-cdf_vals_below)
		else:
			raise ValueError('direction must be "below" or "above"')
		return(cdf_vals)
