#!/usr/bin/python

'''
Reproduces behavior of key matlab functions
'''

import numpy as np

def _bin_centers_to_edges(bin_centers):
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

def hist(x, bins):
	'''
	Reproduces behavior of matlab hist function
	x is an array, and bins is either an integer for the number of
	equally sized bins, or a vector with the centers of unequally
	spaced bins to be used
	Unable to reproduce matlab behavior for an array x with integer
	values on the edges of the bins because matlab behaves
	unpredicatably in those cases
	e.g.: reproduce_matlab_hist(np.array([0, 0, 2, 3, 0, 2]), 3)
	'''
	# if bins is a numpy array, treat it as a list of bin
	# centers, and convert to bin edges
	if isinstance(bins, np.ndarray):
		# identify bin edges based on centers as matlab hist
		# function would
		bin_centers = bins
		bin_edges = _bin_centers_to_edges(bins)
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
