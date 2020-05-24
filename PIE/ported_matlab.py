#!/usr/bin/python

'''
Reproduces behavior of key matlab functions
'''

import cv2
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

def bwperim(img_mask):
	'''
	Reproduces behavior of bwperim with conn = 4 (matlab's default)
	img_mask should be a boolean mask
	Returns a boolean mask of just the contour (perimeter) of img_mask
	'''
	# exact values for second and third arguments here can probably be
	# different (this is the least efficient possible implementation of
	# cv2.findContours), but not haven't tested whether this would
	# impact perimeter identification of actual colonies
	contours = \
		cv2.findContours(np.uint8(img_mask), cv2.RETR_LIST,
			cv2.CHAIN_APPROX_NONE)
	# use contours to find a boolean perim_mask
	perim_mask = cv2.drawContours(np.zeros(np.shape(img_mask)),
		contours[0], -1, 1).astype(bool)
	return(perim_mask)

def bwareaopen(img_mask, P, conn = 8):
	'''
	Reproduces behavior of bwareaopen
	'''
	[label_num, labeled_mask, stat_matrix, _] = \
		cv2.connectedComponentsWithStats(np.uint8(img_mask),
			True, True, False, conn, cv2.CV_32S)
	areas = stat_matrix[:,4]
	# get rows in which areas greater than P
	allowed_labels_with_backgrnd = np.arange(0, label_num)[areas > P]
	# excluding the background (where labeled_objects==0) is key!
	allowed_labels = \
		allowed_labels_with_backgrnd[allowed_labels_with_backgrnd != 0]
	filtered_mask = np.isin(labeled_mask, allowed_labels)
	return(filtered_mask)




