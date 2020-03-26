#!/usr/bin/python

'''
Performs colony edge detection
'''

# !!! TODO: ANALYSIS NOT WORKING ON JPEG?!
# TODO: CHANGE DEFAULT VALUE OF max_proportion_exposed_edge to 0.75 in app

import cv2
import csv
from PIE import adaptive_threshold, ported_matlab
import os
import numpy as np
import pandas as pd
import copy

class _ImageAnalyzer(object):
	'''
	Runs image analysis and creates necessary output files
	'''
	# TODO: save all jpegs as jpeg2000s instead?
	def __init__(self, original_im, image_name, output_path, image_type = 'brightfield',
				hole_fill_area = np.inf, cleanup = False,
				max_proportion_exposed_edge = 0.75,
				save_extra_info = False, threshold_plot_width = 6,
				threshold_plot_height = 4, threshold_plot_dpi = 200,
				threshold_plot_filetype = 'png',
				create_pie_overlay = False):
		# TODO: Reorganize this section to make more sense; add more info about inputs
		# parameter for saving jpegs
		self._jpeg_quality = 98
		# parameters for saving plots
		self._threshold_plot_width = threshold_plot_width
		self._threshold_plot_height = threshold_plot_height
		self._threshold_plot_dpi = threshold_plot_dpi
		self._threshold_plot_filetype = threshold_plot_filetype
		self._save_extra_info = save_extra_info
		self.original_im = original_im
		self.image_type = image_type
		self.hole_fill_area = hole_fill_area
		self.cleanup = cleanup
		self._create_pie_overlay = create_pie_overlay
		self.max_proportion_exposed_edge = max_proportion_exposed_edge
		# get name of image file without path or extension
		self.image_name = image_name
		# set up folders
		self._set_up_folders(output_path)
		# set colors for boundary_im outputs
		self._boundary_color = [255, 0, 255] # magenta
		self._center_color = [145, 224, 47] # bright green
		self._boundary_alpha = 1
		self._cell_center_alpha = 0.75
	
	def _set_up_folders(self, output_path):
		'''
		Sets up necessary folders for image analysis
		'''
		image_output_directories = \
			['jpgGRimages', 'boundary_ims', 'colony_masks', 'threshold_plots',
				'colony_center_overlays', 'single_im_colony_properties']
		self._image_output_dir_dict = dict()
		for current_subdir in image_output_directories:
			current_path = os.path.join(output_path, current_subdir)
			self._image_output_dir_dict[current_subdir] = current_path
			if not os.path.exists(current_path):
				os.makedirs(current_path)

	def _prep_image(self):
		'''
		Reads the image, performs normalization
		If image_type is brightfield, reads as is
		If image_type is phasecontrast, inverts image before analysis
		'''
		# create a normalized image so that assumptions about
		# low-intensity pixel peak being close to 0 made by thresholding
		# algorithm remain true
		# TODO: Think about normalization here. There could be issues
		# related to distance of major true peak from 0 introduced in
		# phase-contrast images if these contain bright points (e.g.
		# white pixels in an otherwise dark brightfield image)
		self.norm_im = \
			cv2.normalize(self.original_im, None, alpha=0, beta=(2**16-1),
				norm_type=cv2.NORM_MINMAX)
		if self.image_type == 'brightfield':
			# the image that was read in is the one that will be
			# processed
			self.input_im = np.copy(self.norm_im)
		elif self.image_type == 'phasecontrast':
			# the normalized image needs to be inverted before
			# processing, then normalized so that assumptions about
			# peak values being close to 0 are true
			self.input_im = cv2.bitwise_not(self.norm_im)
		else:
			raise ValueError(
				"image_type must be either 'brightfield' or 'phasecontrast'")
		

	def _write_threshold_plot(self, threshold_plot):
		'''
		Writes plot of log intensity histogram and threshold to an image
		file
		'''
		if threshold_plot is not None:
			threshold_plot_file = \
				os.path.join(self._image_output_dir_dict['threshold_plots'],
					self.image_name + '_threshold_plot.' + 
					self._threshold_plot_filetype)
		# write threshold plot
		threshold_plot.save(filename = threshold_plot_file,
			width = self._threshold_plot_width,
			height = self._threshold_plot_height, units = 'in',
			dpi = self._threshold_plot_dpi, verbose = False)

	def _write_threshold_info(self, threshold_method_name, threshold):
		'''
		Adds info about current image's threshold and the method used to
		find it to a csv file
		'''
		# !!! NEEDS UNITTEST
		threshold_info_file = \
			os.path.join(self._image_output_dir_dict['threshold_plots'],
					'threshold_info.csv')
		threshold_info = [self.image_name, threshold_method_name, threshold]
		if os.path.exists(threshold_info_file):
			with open(threshold_info_file, 'a') as f:
				writer = csv.writer(f)
				writer.writerow(threshold_info)
		else:
			with open(threshold_info_file, 'w') as f:
				writer = csv.writer(f)
				writer.writerow(threshold_info)

	def _find_cell_centers(self):
		'''
		Find cell centers and save threshold plot if necessary
		'''
		# !!! NEEDS UNITTEST
		self.cell_centers, threshold_method_name, threshold_plot, \
			threshold, default_threshold_method_usage = \
			adaptive_threshold.threshold_image(self.input_im,
				self._save_extra_info)
		# always save 'extra info' (boundary ims, threshold plots,
		# combined cell center/boundary images) if default
		# threshold_method was not used
		if not default_threshold_method_usage:
			self._save_extra_info = True
		# save threshold plot and info if necessary
		if self._save_extra_info:
			self._write_threshold_plot(threshold_plot)
			self._write_threshold_info(threshold_method_name, threshold)

	def _find_colony_mask(self):
		'''
		Runs PIE edge detection and finds the binary mask
		'''
		# !!! NEEDS UNITTEST
		pie_edge_detector = _EdgeDetector(self.input_im, self.cell_centers,
			self.hole_fill_area, self.cleanup, self.max_proportion_exposed_edge)
		self.colony_mask = pie_edge_detector.run_edge_detection()

	def _save_jpeg(self):
		'''
		Writes jpeg of original input image
		'''
		# !!! NEEDS UNITTEST
		jpeg_path = os.path.join(self._image_output_dir_dict['jpgGRimages'],
			self.image_name + '.jpg')
		# check whether original_im is > 8-bit; if so, decrease bitdepth
		jpeg_im = copy.copy(self.original_im)
		while np.max(jpeg_im) > 2**8:
			jpeg_im = jpeg_im / 2**8
		cv2.imwrite(jpeg_path, jpeg_im,
			[cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])

	def _save_colony_mask(self):
		'''
		Saves colony mask as 8-bit tif file
		'''
		# !!! NEEDS UNITTEST
		colony_mask_path = os.path.join(self._image_output_dir_dict['colony_masks'],
			self.image_name + '.tif')
		cv2.imwrite(colony_mask_path, np.uint8(self.colony_mask)*255)

	def _save_boundary_im(self):
		'''
		Creates and saves an image in which the normalized image is
		overlaid with a boundary (contour) of the colony mask
		'''
		colony_boundary_mask = ported_matlab.bwperim(self.colony_mask)
		norm_im_8_bit = self.norm_im/(2**8)
		self.boundary_im = create_color_overlay(norm_im_8_bit,
			colony_boundary_mask, self._boundary_color, self._boundary_alpha)
		boundary_im_path = \
			os.path.join(self._image_output_dir_dict['boundary_ims'],
				self.image_name + '.jpg')
		cv2.imwrite(boundary_im_path, self.boundary_im,
			[cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])

	def _save_colony_center_overlay(self):
		'''
		Creates and saves an image in which the normalized image is
		overlaid with a boundary (contour) of the colony mask, as well
		as the cell center mask
		'''
		self.colony_center_overlay = create_color_overlay(self.boundary_im,
			self.cell_centers, self._center_color, self._cell_center_alpha)
		colony_center_overlay_path = \
			os.path.join(self._image_output_dir_dict['colony_center_overlays'],
				self.image_name + '.jpg')
		cv2.imwrite(colony_center_overlay_path, self.colony_center_overlay,
			[cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])

	def _save_colony_properties(self):
		'''
		Measures and records area and centroid in every detected colony
		'''
		# !!! NEEDS UNITTEST !!!
		colony_property_finder = _ColonyPropertyFinder(self.colony_mask)
		self.colony_property_df = colony_property_finder.get_properties()
		colony_property_path = \
			os.path.join(self._image_output_dir_dict['single_im_colony_properties'],
				self.image_name + '.csv')
		colony_property_df_to_save = self.colony_property_df[['Area', 'Perimeter', 'cX', 'cY']]
		colony_property_df_to_save.to_csv(colony_property_path, index = False)

	def _save_post_edge_detection_files(self):
		'''
		Saves colony mask, jpeg, regionprops, and if necessary, boundary_ims, etc
		'''
		# write input image as a jpeg
		self._save_jpeg()
		# write colony mask as 8-bit tif file
		self._save_colony_mask()
		# measure and save colony properties
		self._save_colony_properties()
		# if self._save_extra_info is True, save additional files
		if self._save_extra_info:
			self._save_boundary_im()
			self._save_colony_center_overlay()

	def process_image(self):
		'''
		Performs all image processing
		'''
		# !!! NEEDS UNITTEST ???
		# read in input image
		self._prep_image()
		# identify cell centers
		self._find_cell_centers()
		# identify colony mask
		self._find_colony_mask()
		# save analysis output files
		self._save_post_edge_detection_files()
		return(self.colony_mask, self.colony_property_df)

class _ColonyPropertyFinder(object):
	'''
	Measures and holds properties of colonies based on a colony mask
	'''
	# !!! TODO: CHECK WHICH OTHER PARAMETERS ARE NEEDED DOWNSTREAM, RECORD THOSE TOO
	def __init__(self, colony_mask):
		self.colony_mask = colony_mask
		self.property_df = pd.DataFrame()

	def _find_connected_components(self):
		'''
		Finds connected components in a colony
		'''
		[self._label_num, self._labeled_mask, self._stat_matrix, self._centroids] = \
			cv2.connectedComponentsWithStats(np.uint8(self.colony_mask),
				True, True, True, 4, cv2.CV_32S)

	def _find_areas(self):
		'''
		Records the areas of each colony
		'''
		areas = self._stat_matrix[:,4]
		# remove background
		self.property_df['Area'] = areas[1:]

	def _find_centroids(self):
		'''
		Records centroids of each colony
		'''
		# remove background centroid
		self.property_df['cX'] = self._centroids[1:,0]
		self.property_df['cY'] = self._centroids[1:,1]

	def _find_bounding_box(self):
		'''
		Records bounding box of each colony
		'''
		# !!! NEEDED???
		# Can be found in self._stat_matrix
		pass

	def _find_flat_coordinates(self, single_colony_mask):
		'''
		Finds indices of True pixels in flattened single_colony_mask
		Returns a string of indices concatenated separated by a space
		(these can be read back into np array via np.fromstring)
		'''
		# indices in flattened current_mask
		# can be used to reference indices in original array as
		# current_mask.flat(np.fromstring(concat_coords,
		#	dtype = int, sep = ' '))
		flat_coords = np.flatnonzero(single_colony_mask)
		# concatenate indices with space
		concat_coords = ' '.join([str(k) for k in flat_coords])
		return(concat_coords)

	def _find_perimeter(self, single_colony_mask):
		'''
		Finds external perimeter of mask that should contain single
		colony
		NB: perimeter calculated here between centers of consecutive
		contour pixels
		'''
		colony_cont = \
			cv2.findContours(np.uint8(single_colony_mask), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_NONE)[0][0]
		colony_perim = cv2.arcLength(colony_cont, True)
		return(colony_perim)

	def _find_colonywise_properties(self):
		'''
		Loops through detected colonies and measures the properties that
		need to be measured one at a time
		'''
		self.property_df['Perimeter'] = np.nan
		self.property_df['PixelIdxList'] = None
		# don't loop through background
		for colony in range(1, self._label_num):
			current_colony_mask = self._labeled_mask == colony
			self.property_df.at[colony-1, 'Perimeter'] = \
				self._find_perimeter(current_colony_mask)
			self.property_df.at[colony-1, 'PixelIdxList'] = \
				self._find_flat_coordinates(current_colony_mask)

	def _measure_and_record_properties(self):
		'''
		Measures and records all colony properties
		'''
		self._find_connected_components()
		self._find_areas()
		self._find_centroids()
		self._find_colonywise_properties()

	def get_properties(self):
		'''
		Returns dataframe of colony properties
		'''
		self._measure_and_record_properties()
		return(self.property_df)

class _EdgeDetector(object):
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

	def draw_pie_pieces(self):
		# TODO: write this method
		pass

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
	If keep_overlapping_objects is True, returns a mask containing only
	those objects that overlap guide_mask
	If keep_overlapping_objects is False, returns a mask containing all
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
	# Identify the positions in each labeled_objects that overlap
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

def create_color_overlay(image, mask, mask_color, mask_alpha):
	'''
	Creates an rbg image of image with mask overlaid on it in
	mask_color with mask_alpha
	mask_color is a list of r, g, b values on a scale of 0 to 255
	'''
	# TODO: Doesn't work when image has a max of 0 (because of np.max(color_image) issue)
	# check whether image is grayscale; if so, convert to rgb
	color_image = copy.copy(image)
	if len(np.shape(color_image)) == 2:
		color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
	# adjust mask_color by alpha, inverse, scale to image bitdepth
	mask_color_adjusted_tuple = \
		tuple(float(k)/255*np.max(color_image)*mask_alpha for k in mask_color[::-1])
	color_image[mask] = \
		np.round(color_image[mask].astype(float) * (1-mask_alpha) +
			mask_color_adjusted_tuple).astype(int)
	return(color_image)

def read_and_run_analysis(input_im_path, output_path,
	image_type, hole_fill_area, cleanup, max_proportion_exposed_edge,
	save_extra_info = True):
	'''
	Reads image from input_im_path and runs PIE colony detection on it,
	saving required files to output_path
	'''
	image = cv2.imread(input_im_path, cv2.IMREAD_ANYDEPTH)
	image_name = os.path.splitext(os.path.basename(input_im_path))[0]
	image_analyzer = _ImageAnalyzer(image, image_name, output_path,
		image_type, hole_fill_area, cleanup, max_proportion_exposed_edge,
		save_extra_info)
	image_analyzer.process_image()
	return(image_analyzer.colony_mask, image_analyzer.colony_property_df)

# !!! NB: explain in doc that this assumes that there will never be two images with the same name being placed in the same output folder


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