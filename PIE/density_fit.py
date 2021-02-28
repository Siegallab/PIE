#!/usr/bin/python

'''
Contains abstract class for fitting density
'''

import numpy as np
import sys
import warnings
from scipy.optimize import least_squares, minimize, Bounds, LinearConstraint, \
	NonlinearConstraint, basinhopping, shgo, dual_annealing

class _DensityFitter(object):
	'''
	Abstract class for fitting densities

	Inheriting classes need:
	-	self.param_idx_dict, which contains index of every parameter
		name
	-	self.non_neg_params, which contain names of every parameter
		that must be => 0
	-	self.above_zero_params, which contain names of every parameter
		that must be > 0
	'''

	def _check_bounds(self, bounds):
		'''
		Sets bounds on parameters
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
				str(len(self.param_idx_dict)))
		return(bounds)

	def _id_starting_vals(self):
		'''
		Identifies starting values for parameters

		Creates self.starting_param_vals, a numpy array of starting
		values with indices corresponding to those in
		self.param_idx_dict
		'''
		pass

	def _check_starting_vals(self):
		'''Check that starting parameters between bounds'''
		lower_bound_fail = self.starting_param_vals < self.lower_bounds
		upper_bound_fail = self.starting_param_vals > self.upper_bounds
		self.starting_param_vals[lower_bound_fail] = \
			self.lower_bounds[lower_bound_fail]
		self.starting_param_vals[upper_bound_fail] = \
			self.lower_bounds[upper_bound_fail]

	def fit_density(self, *args):
		'''
		Finds best parameter values to fit y_vals

		Runs _generate_fit_result_dict to make a dict of parameters
		that contains fit results
		'''
		pass

	def _generate_fit_result_dict(self):
		'''
		Generates a dict of parameters that contains fit results
		'''
		self.fit_result_dict = {param_name: self.fit_results.x[param_idx]
			for param_name, param_idx in self.param_idx_dict.items()}

	def plot(self):
		'''
		Creates plot of data and fit
		'''
		pass

class DensityFitterMLE(_DensityFitter):
	'''
	Abstract class for fitting densities using Maximum Likelihood
	Estimation

	Inheriting classes need:
	-	self.param_idx_dict, which contains index of every parameter
		name
	-	self.data, which contains datapoints for MLE

	If calling bounds, requires:
	-	self.lower_bounds and self.upper_bounds
	-	self.non_neg_params, which contain names of every parameter
		that must be => 0
	-	self.above_zero_params, which contain names of every parameter
		that must be > 0

	Can also include constraints on the MLE function as
	self.linear_constraints and self.nonlinear_constraints
	'''

	def _neg_ll_fun(self, params, data):
		'''
		Calculates negative log likelihood of observing data given
		params
		'''
		pass

	def _define_constrains(self):
		'''
		Return constraint and bound objects
		'''
		# check for bounds
		if (self.lower_bounds is None or 
			np.all(self.lower_bounds == -np.inf)) and \
			(self.upper_bounds is None or 
				np.all(self.upper_bounds == np.inf)):
			# no bounds
			bound_obj = None
		else:
			bound_obj = Bounds(self.lower_bounds, self.upper_bounds)
		constraints = []
		if self.linear_constraints is not None:
			constraints.append(LinearConstraint(self.linear_constraints))
		if self.nonlinear_constraints is not None:
			constraints.append(NonlinearConstraint(self.nonlinear_constraints))
		if len(constraints)==1:
			constraints = constraints[0]
		return(bound_obj, constraints)

	def _accept_test(self, **kwargs):
		'''
		accept_test for global minimization
		'''
		x = kwargs['x_new']
		tmax = bool(np.all(x <= self.upper_bounds))
		tmin = bool(np.all(x >= self.lower_bounds))
		return tmax and tmin

	def fit_density(self, opt_method = None, min_type = 'global'):
		'''
		Finds best parameter values to fit data

		Generates a dict of parameters that contains fit results
		'''
		bound_obj, constraints = self._define_constrains()
		if bound_obj is not None or len(constraints)>0:
			# do constrained fit
			if opt_method is not None and opt_method != 'trust-constr':
				raise ValueError(
					'Only trust-constr opt_method can be used for constrained '
					'optimization')
			else:
				opt_method = 'trust-constr'
#				opt_method = 'SLSQP'
		else:
			# resort to Nelder-Mead
			if opt_method is None:
				opt_method = 'Nelder-Mead'
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			if min_type == 'local':
				self.fit_results = minimize(
					self._neg_ll_fun,
					self.starting_param_vals,
					method = opt_method,
					bounds = bound_obj,
					constraints = constraints)
			elif min_type == 'global':
				self.fit_results = dual_annealing(
					self._neg_ll_fun,
					list(zip(self.lower_bounds, self.upper_bounds)),
					x0 = self.starting_param_vals,
					local_search_options = {
						'method':opt_method,
						'constraints':constraints
						})
			else:
				raise ValueError('min_type must be "global" or "local"')
		self._generate_fit_result_dict()

class DensityFitterLS(_DensityFitter):
	'''
	Abstract class for fitting densities using least squares

	Inheriting classes need:
	-	self.param_idx_dict, which contains index of every parameter
		name
	-	self.non_neg_params, which contain names of every parameter
		that must be => 0
	-	self.above_zero_params, which contain names of every parameter
		that must be > 0
	'''

	def _residual_fun(self, params):
		'''
		Calculates residuals of difference between
		density(data_df.x) and data_df.y
		'''
		pass
		
	def fit_density(self):
		'''
		Finds best parameter values to fit y_vals

		Generates a dict of parameters that contains fit results
		'''
		self.fit_results = \
			least_squares(self._residual_fun,
				self.starting_param_vals,
				bounds = (self.lower_bounds, self.upper_bounds))
		self.data['y_hat'] = self.data.y - self.fit_results.fun
		self._generate_fit_result_dict()
