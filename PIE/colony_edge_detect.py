#!/usr/bin/python

'''
Performs colony edge detection
'''

# !!! TODO: ANALYSIS NOT WORKING ON JPEG?!

import cv2
import os
import numpy as np
import pandas as pd
from PIE import ported_matlab, image_coloring
from PIL import ImageColor

class EdgeDetector(object):
	'''
	Detects edges in input_im using PIE algorithm
	'''

	def __init__(self, input_im, cell_centers, hole_fill_area, cleanup,
		max_proportion_exposed_edge):
		'''
		Read input_im_path as grayscale image of arbitrary bitdepth
		'''
		self.input_im = input_im
		self.cell_centers = cell_centers
		self.hole_fill_area = hole_fill_area
		self.cleanup = cleanup
		self.max_proportion_exposed_edge = max_proportion_exposed_edge
		# create 'pie' quadrants
		self.pie_piece_position_dict = \
			{'i': np.array([[0, 1], [0, 0]]),
			'ii': np.array([[1, 0], [0, 0]]),
			'iii': np.array([[0, 0], [1, 0]]),
			'iv': np.array([[0, 0], [0, 1]])}
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
		# TODO: This leaves empty spots at the intensity peaks and
		# troughs; > should really be >= instead (or < should be <=)
		Gx_right = Gx < 0
		Gx_left = Gx > 0
		Gy_top = Gy > 0
		Gy_bottom = Gy < 0
		self.pie_piece_dict = dict()
		self.pie_piece_dict['i'] = _PiePiece(Gx_right, Gy_top)
		self.pie_piece_dict['ii'] = _PiePiece(Gx_left, Gy_top)
		self.pie_piece_dict['iii'] = _PiePiece(Gx_left, Gy_bottom)
		self.pie_piece_dict['iv'] = _PiePiece(Gx_right, Gy_bottom)

	def _create_inital_colony_mask(self):
		'''
		Runs initial edge detection without cleanup
		'''
		# identify 'pie pieces' that overlap cell centers, and combine
		# them into a composite colony_mask
		initial_colony_mask = np.zeros(self.input_im.shape, dtype = bool)
		for pie_quad_name, pie_piece_quadrant in \
			self.pie_piece_dict.items():
			pie_piece_quadrant.id_center_overlapping_pie_pieces(
					self.cell_centers)
			initial_colony_mask = np.logical_or(initial_colony_mask,
				pie_piece_quadrant.cell_overlap_pie_mask)
		return(initial_colony_mask)

	def _fill_holes(self, colony_mask):
		'''
		Fills holes in colony_mask that are smaller than
		self.hole_fill_area
		(Does this by inverting colony mask and removing
		objects with area above self.hole_fill_area;
		Assumes background is the largest 'hole' (0-filled area) in
		colony_mask)
		'''
		inverted_colony_mask = np.invert(colony_mask)
		# note that connectivity here needs to be 8 (for background
		# pixels) in order for effective connectivity for foreground
		# objects to be 4
		[label_num, labeled_mask, stat_matrix, _] = \
			cv2.connectedComponentsWithStats(np.uint8(inverted_colony_mask),
				True, True, False, 8, cv2.CV_32S)
		areas = stat_matrix[:,4]
		# get rows in which areas greater than P
		allowed_labels_with_backgrnd = \
			np.arange(0, label_num)[areas > self.hole_fill_area]
		# excluding the background (where labeled_objects==0) is key!
			# (this is really foreground from colony_mask)
		# in order for this algorithm to work when self.hole_fill_area
		# is large (e.g. np.inf), need to also detect what used to be
		# background (but not holes) in colony_mask and remove that from
		# allowed_labels too
		# to do this, assume the largest object in inverted_colony_matrix
		# is background
		background_label = np.argmax(areas[1:]) + 1
		allowed_labels = \
			allowed_labels_with_backgrnd[
				np.logical_and((allowed_labels_with_backgrnd != 0),
					(allowed_labels_with_backgrnd != background_label))]
		filtered_mask = np.isin(labeled_mask, allowed_labels)
		filled_holes_mask = np.invert(filtered_mask)
		# put original background back in
		filled_holes_mask[labeled_mask == background_label] = False
		return(filled_holes_mask)

	def _clear_mask_edges(self, mask_to_clear):
		'''
		Removes any objects in mask_to_clear that touch its edge
		'''
		mask_edge = \
			ported_matlab.bwperim(np.ones(np.shape(mask_to_clear), dtype = bool))
		cleared_mask, _ = \
			_filter_by_overlaps(mask_to_clear, mask_edge,
				keep_overlapping_objects = False)
		return(cleared_mask)

	def _create_translation_matrix(self, source_position_mat,
		target_position_mat):
		'''
		Returns a translation matrix that can be used with
		cv2.warpAffine that moves an image from source_position_mat to
		target_position_mat
		'''
		row_col_translation = \
			np.subtract(np.where(target_position_mat),
				np.where(source_position_mat)).squeeze()
		xy_translation = np.flip(row_col_translation, 0)
		translation_mat = np.array([[1, 0, xy_translation[0]],
			[0, 1, xy_translation[1]]])
		return(translation_mat.astype(float))

	def _perform_neighbor_filtering(self, focal_pie_quad_name,
		focal_pie_piece_quadrant):
		'''
		Filter objects in focal_pie_piece_quadrant that have correct
		neighbors
		'''
		# loop through potential neighboring pie pieces
		for neighbor_pie_quad_name, neighbor_pie_piece_quadrant in \
			self.pie_piece_dict.items():
			# get translation matrix that moves an image in the
			# 'neighbor' pie piece's quadrant by 1 pixel into
			# the 'focal' pie piece's quadrant
			translation_mat = \
				self._create_translation_matrix(
					self.pie_piece_position_dict[
						neighbor_pie_quad_name],
					self.pie_piece_position_dict[focal_pie_quad_name])
			# filter focal pie pieces based on whether they
			# overlap with this neighbor in the expected
			# direction
			focal_pie_piece_quadrant.filter_by_neighbor(
				neighbor_pie_piece_quadrant.edge_filtered_pie_mask,
				translation_mat)

	def _single_round_edge_filtering(self, colony_mask_previous_holes):
		'''
		Peforms one round of edge filtering on all pie pieces
		'''
		for pie_quad_name, pie_piece_quadrant in \
			self.pie_piece_dict.items():
			pie_piece_quadrant.filter_by_exposed_edge(
				colony_mask_previous_holes,
				self.max_proportion_exposed_edge)

	def _single_round_neighbor_filtering(self, combined_filtered_pieces):
		'''
		loop through edge-filtered pie pieces and identify pie pieces
		that have the correct neighbors from two directions (e.g. pie
		piece i should neighbor iii from below and ii from the left)
		'''
		for focal_pie_quad_name, focal_pie_piece_quadrant in \
			self.pie_piece_dict.items():
			# keep objects in
			# focal_pie_piece_quadrant.neighbor_filtered_pie_mask
			# that have correct neighbors
			self._perform_neighbor_filtering(focal_pie_quad_name,
				focal_pie_piece_quadrant)
			# combine filtered pie pieces into single mask
			combined_filtered_pieces  = \
				np.logical_or(combined_filtered_pieces,
					focal_pie_piece_quadrant.neighbor_filtered_pie_mask)
		return(combined_filtered_pieces)

	def _run_cleanup(self, initial_colony_mask, colony_mask_filled_holes):
		'''
		Runs cleanup procedure
		'''
		# Produces different results than original matlab script
		# Matlab script only considered 'holes' in the colony mask that
		# were present since before cleanup, so it ignored any expansion
		# of those holes (including linking them to background) that
		# resulted from the cleanup; current version of algorithm does
		# not do that
		colony_mask_previous = np.copy(colony_mask_filled_holes)
		colony_mask_previous_holes = np.copy(initial_colony_mask)
		mask_diff = 1
		while mask_diff > 0:
			combined_filtered_pieces = \
				np.zeros(self.input_im.shape, dtype = bool)
			# loop through pie pieces and create edge-filtered pie pieces
			self._single_round_edge_filtering(colony_mask_previous_holes)
			# loop through edge-filtered pie pieces	and create neighbor-
			# filtered pie pices
			combined_filtered_pieces = \
				self._single_round_neighbor_filtering(combined_filtered_pieces)
			# copy mask from filtered pie pieces as starting point for
			# filtering in the next round
			colony_mask_previous_holes = np.copy(combined_filtered_pieces)
			colony_mask_filled_holes_filt = \
				self._fill_holes(colony_mask_previous_holes)
			mask_diff = np.sum(np.logical_xor(colony_mask_filled_holes_filt,
				colony_mask_previous))
			colony_mask_previous = np.copy(colony_mask_filled_holes_filt)
		return(colony_mask_filled_holes_filt)

	def run_edge_detection(self):
		'''
		Runs full edge detection and returns colony mask
		'''
		initial_colony_mask = self._create_inital_colony_mask()
		colony_mask_filled_holes = self._fill_holes(initial_colony_mask)
		if self.cleanup:
			colony_mask_filled_holes = \
				self._run_cleanup(initial_colony_mask, colony_mask_filled_holes)
		colony_mask_clear_edges = \
			self._clear_mask_edges(colony_mask_filled_holes)
		return(colony_mask_clear_edges)

	def draw_pie_pieces(self,
		mask_type = 'cell_overlap_pie_mask',
		color_list = ['#75D5E6','#A10279','#FFE53B','#3B5B9B'],
		background_im = None,
		background_im_bitdepth = None,
		pie_shading_alpha = 1
		):
		'''
		Returns image with each pie piece show in one of the colors in 
		color_list

		mask_type is the type of pie mask to use; options are 
		'pie_mask' (input image broken up into pie pieces), 
		'cell_overlap_pie_mask' (mask overlapping with cell centers), 
		'edge_filtered_pie_mask' (pie pieces after latest filtration 
		of exposed edges), or 'neighbor_filtered_pie_mask' (pie pieces 
		after latest filtration based on neighboring pie piece 
		quadrants)

		color_list can either be a string (to use same color for all 
		PIE pieces) or a list of 4 colors as stings (either web color 
		names or hex keys); default is colorblind-friendly

		background_im can be an image or None(default), in which case 
		pie pieces will have black background

		pie_shading_alpha is the opacity of the pie pieces, on a scale 
		of 0 to 1
		'''
		# !!! NEEDS UNITTEST
		# do qc on color_list
		error_text = \
			'color_list must be a string or a list of color names with length 4'
		if isinstance(color_list, list):
			if len(color_list) == 1:
				color_list = color_list*4
			elif len(color_list) != 4:
				raise IndexError(error_text)
		elif isinstance(color_list, str):
			color_list = [color_list]*4
		else:
			raise TypeError(error_text)
		# convert color_list to rgb
		rgb_color_vals = [ImageColor.getcolor(c, "RGB") for c in color_list]
		rgb_color_dict = dict(zip(self.pie_piece_dict.keys(), rgb_color_vals))
		# loop over pie pieces and color each one on background_im
		if background_im is None:
			colored_im = np.uint8(np.zeros(self.input_im.shape))
			background_im_bitdepth = 8
		else:
			colored_im = background_im.copy()
		for pie_quad, pie_piece_quadrant in self.pie_piece_dict.items():
			pie_mask = getattr(pie_piece_quadrant, mask_type)
			# here we're relying on pie pieces not overlapping to just 
			# do this simply and sequentially
			colored_im = image_coloring.create_color_overlay(
				colored_im,
				pie_mask,
				rgb_color_dict[pie_quad],
				pie_shading_alpha,
				bitdepth = background_im_bitdepth
				)
		return(colored_im)

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
		self.cell_overlap_pie_mask, \
			self.cell_overlap_labeled_pie_mask = \
			_filter_by_overlaps(self.pie_mask, cell_center_mask,
				keep_overlapping_objects = True)

	def filter_by_exposed_edge(self, colony_mask, max_proportion_exposed_edge):
		'''
		Identify pie_pieces for which the proportion of the contour that
		is part of the contour of colony_mask does not exceed
		max_proportion_exposed_edge
		'''
		# Identified pie pieces that are part of colony_mask
		pie_pieces = \
			np.logical_and(self.cell_overlap_pie_mask, colony_mask)
		# Create mask with the contour of each pie piece overlapping
		# colony_mask
		pie_piece_perim = ported_matlab.bwperim(pie_pieces)
		# Create mask with the contour of each complete colony
		colony_perim = ported_matlab.bwperim(colony_mask)
		# Create images with the parts of the perimeter of each pie
		# piece that overlaps with the outer edge of the object
		exposed_pie_edges = \
			np.logical_and(pie_piece_perim, colony_perim)
		exposed_pie_edge_labeled = self.cell_overlap_labeled_pie_mask * \
			exposed_pie_edges.astype(int)
		total_pie_piece_perim_labeled = self.cell_overlap_labeled_pie_mask * \
			pie_piece_perim.astype(int)
		# make a list of potential non-background pie piece labels (with
		# an extra np.inf at the end so that np.histogram doesn't bin
		# the counts for the last two values
		pie_piece_labels_with_background = \
			np.unique(total_pie_piece_perim_labeled)
		pie_piece_labels = \
			pie_piece_labels_with_background[
				pie_piece_labels_with_background > 0]
		pie_piece_histogram_bins = np.append(pie_piece_labels, np.inf)
		# Count total perimeter pixels of every pie piece
		pie_piece_tot_border_counts, _ = \
			np.histogram(total_pie_piece_perim_labeled,
				pie_piece_histogram_bins)
		# Count the perimeter pixels of every pie piece that overlaps
		# with the outer edge of colony_mask (i.e., which is an external
		# edge in a colony)
		pie_piece_exposed_border_counts, _ = \
			np.histogram(exposed_pie_edge_labeled, pie_piece_histogram_bins)
		# calculate the proportion of each pie piece edge that is
		# 'exposed' (on the outer edge of the colony)
		pie_piece_exposed_border_proportion = \
			np.divide(pie_piece_exposed_border_counts.astype(float),
				pie_piece_tot_border_counts.astype(float))
		# only allow objects with 
		allowed_objects = pie_piece_labels[
			pie_piece_exposed_border_proportion <=
				max_proportion_exposed_edge]
		self.edge_filtered_pie_mask = \
			np.isin(self.cell_overlap_labeled_pie_mask, allowed_objects)

	def filter_by_neighbor(self, neighbor_pie_mask, neighbor_translation_mat):
		'''
		Check whether neighbor_pie_mask is a horizontal/vertical
		neighbor (rather than diagonal neighbor, or this pie piece
		itself)
		'''
		# if no neighbor_filtered_pie_mask created yet, make one
		if not hasattr(self, 'neighbor_filtered_pie_mask'):
			self.neighbor_filtered_pie_mask = np.copy(self.edge_filtered_pie_mask)
		if np.sum(np.abs(neighbor_translation_mat[:,2])) == 1:
			translated_neighbor_pie_mask = \
				cv2.warpAffine(np.uint8(neighbor_pie_mask),
					neighbor_translation_mat, (neighbor_pie_mask.shape[1],
						neighbor_pie_mask.shape[0])).astype(bool)
			# only keep those pie pieces in the focal pie piece that are
			# neighbored by the 'neighbor' pie piece in the expected
			# direction
			neighbor_overlap_mask = \
				np.logical_and(self.neighbor_filtered_pie_mask,
					translated_neighbor_pie_mask)
			self.neighbor_filtered_pie_mask, _ = \
				_filter_by_overlaps(self.neighbor_filtered_pie_mask,
					neighbor_overlap_mask, True,
					self.cell_overlap_labeled_pie_mask)

def _filter_by_overlaps(mask_to_filter, guide_mask, keep_overlapping_objects,
	labeled_mask = None):
	'''
	Identifies objects in binary mask_to_filter that overlap with binary
	guide_mask

	If keep_overlapping_objects is True, returns a mask containing 
	only those objects that overlap guide_mask; if 
	keep_overlapping_objects is False, returns a mask containing all 
	objects except the ones that overlap guide_mask

	If labeled_mask is not None, uses supplied labeled_mask
	'''
	# !!! NEEDS UNITTEST
	if labeled_mask is None:
		# Label each contiguous object in the image with a unique int
		# When identifying the objects, we must not count diagonal
		# neighbor pixels as belonging to the same object (this is VERY
		# important)
		# Default for cv2.connectedComponents is connectivity = 8
		label_num, labeled_mask = \
			cv2.connectedComponents(np.uint8(mask_to_filter), connectivity = 4)
	else:
		### !!! NEEDS UNITTEST!
		# use supplied labeled_mask
		label_num = np.max(labeled_mask)+1
	# Identify the elements in each labeled_objects that overlap
	# with guide_mask
	guide_overlap = labeled_mask[guide_mask]
	# Identify the label numbers (allowed_objects) that overlap with
	# cell_center_mask
	overlapping_labels_with_backgrnd = np.unique(guide_overlap)
	if keep_overlapping_objects:
		allowed_labels_with_backgrnd = overlapping_labels_with_backgrnd
	else:
		# find objects that are NOT in overlapping_labels_with_backgrnd
		allowed_labels_with_backgrnd = \
			np.setdiff1d(np.arange(0, label_num),
				overlapping_labels_with_backgrnd)
	# excluding the background (where labeled_objects==0) is key!
	allowed_labels = \
		allowed_labels_with_backgrnd[allowed_labels_with_backgrnd != 0]
	# identify pixels in labeled_pie_mask whose labels are in
	# allowed_labels
	filtered_mask = \
		np.isin(labeled_mask, allowed_labels)
	return(filtered_mask, labeled_mask)

if __name__ == '__main__':
	pass