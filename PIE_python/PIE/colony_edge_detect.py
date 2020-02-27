#!/usr/bin/python

'''
Performs colony edge detection
'''

import cv2
from PIE import adaptive_threshold
import os
from PIE import ported_matlab
import numpy as np

class _ImageAnalyzer(object):
	'''
	Runs image analysis and creates necessary output files
	'''

	def __init__(self, input_im_path, output_path, image_type = 'brightfield',
				hole_fill_area = np.inf, cleanup = False,
				max_proportion_exposed_edge = 0.25,
				save_extra_info = False, threshold_plot_width = 6,
				threshold_plot_height = 4, threshold_plot_dpi = 200,
				threshold_plot_filetype = 'jpg'):
		# parameter for saving jpegs
		self._jpeg_quality = 95
		# parameters for saving plots
		self._threshold_plot_width = threshold_plot_width
		self._threshold_plot_height = threshold_plot_height
		self._threshold_plot_dpi = threshold_plot_dpi
		self._threshold_plot_filetype = threshold_plot_filetype
		self._save_extra_info = save_extra_info
		self.input_im_path = input_im_path
		self.image_type = image_type
		self.cleanup = cleanup
		self.max_proportion_exposed_edge = max_proportion_exposed_edge
		# get name of image file without path or extension
		self.image_name = os.path.splitext(os.path.basename(input_im_path))[0]
		# set up folders
		self._set_up_folders(output_path)
	
	def _set_up_folders(self, output_path):
		'''
		Sets up necessary folders for image analysis
		'''
		# !!! NEEDS UNITTEST
		image_output_directories = \
			['jpgGRimages', 'boundary_ims', 'colony_masks', 'threshold_plots',
				'colony_center_overlays']
		self._image_output_dir_dict = dict()
		for current_subdir in image_output_directories:
			current_path = os.path.join(output_path, current_subdir)
			self.image_output_dir_dict[current_subdir] = current_path
			if not os.path.exists(current_path):
				os.makedirs(current_path)

	def _read_image(self, input_im_path, image_type):
		'''
		Reads the image
		If image_type is brightfield, reads as is
		If image_type is phasecontrast, inverts image before analysis
		'''
		# !!! NEEDS UNITTEST
		# read in original image
		self.original_im = cv2.imread(input_im_path, cv2.IMREAD_ANYDEPTH)
		if self.image_type == 'brightfield':
			# the image that was read in is the one that will be
			# processed
			self.input_im = self.original_im
		elif self.image_type == 'phasecontrast':
			# the image that was read in needs to be inverted before
			# processing
			self.input_im = cv2.bitwise_not(self.original_im)
		else:
			raise ValueError(
				"image_type must be either 'brightfield' or 'phasecontrast'")
		# create a normalized image for display purposes
		self.norm_im = \
			cv2.normalize(self.original_im, None, alpha=0, beta=1,
				norm_type=cv2.NORM_MINMAX)

	def _save_jpeg(self):
		'''
		Writes jpeg of original input image
		'''
		# !!! NEEDS UNITTEST
		jpeg_path = os.path.join(self._image_output_dir_dict('jpgGRimages'),
			self.image_name + '.jpg')
		cv2.imwrite(jpeg_path, self.original_im,
			[cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])

	def _write_threshold_plot(self, threshold_plot):
		'''
		Writes plot of log intensity histogram and threshold to an image
		file
		'''
		# !!! NEEDS UNITTEST
		if threshold_plot is not None:
			threshold_plot_file = \
				os.path.join(self._image_output_dir_dict('threshold_plots'),
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
			os.path.join(self._image_output_dir_dict('threshold_plots'),
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
				self.save_extra_info)
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

	def _save_post_edge_detection_files(self):
		'''
		Saves overlays, regionprops, and if necessary, bwperim, etc
		'''

	def process_image(self):
		'''
		Performs all image processing
		'''
		# !!! NEEDS UNITTEST
		# read in input image
		self._read_image(self.input_im_path, self.image_type)
		# write input image as a jpeg
		self._save_jpeg()
		# identify cell centers
		self._find_cell_centers()
		# identify colony mask
		self._find_colony_mask()
		# save analysis output files
		self._save_post_edge_detection_files()

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

	def _create_inital_colony_mask(self):
		'''
		Runs initial edge detection without cleanup
		'''
		### ??? NEEDS UNITTEST ??? ###
		# identify 'pie pieces' that overlap cell centers, and combine
		# them into a composite colony_mask
		initial_colony_mask = np.zeros(self.input_im.shape, dtype = bool)
		for pie_quad_name, pie_piece_quadrant in \
			self.pie_piece_dict.iteritems():
			current_cell_pieces = \
				pie_piece_quadrant.id_center_overlapping_pie_pieces(
					self.cell_centers)
			initial_colony_mask = np.logical_or(initial_colony_mask,
				current_cell_pieces)
		return(initial_colony_mask)

	def _fill_holes(self, colony_mask):
		'''
		Fills holes in colony_mask by inverting colony mask and removing
		objects with area above self.hole_fill_area
		Assumes background is the largest 'hole' (0-filled area) in
		colony_mask
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
		### !!! NEEDS UNITTEST
		mask_edge = \
			ported_matlab.bwperim(np.ones(np.shape(mask_to_clear), dtype = bool))
		cleared_mask = \
			_filter_by_overlaps(mask_to_clear, mask_edge,
				keep_overlapping_objects = False)
		return(cleared_mask)

	def run_edge_detection(self):
		'''
		Runs full edge detection and returns colony mask
		'''
		### !!! NEEDS UNITTEST
		initial_colony_mask = self._create_inital_colony_mask()
		colony_mask_filled_holes = self._fill_holes(initial_colony_mask)
		if self.cleanup:
			raise RuntimeError('cleanup mode is not yet complete')
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
		cell_overlapping_pie_mask = \
			_filter_by_overlaps(self.pie_mask, cell_center_mask,
				keep_overlapping_objects = True)
		return(cell_overlapping_pie_mask)



def _filter_by_overlaps(mask_to_filter, guide_mask, keep_overlapping_objects):
	'''
	Identifies objects in binary mask_to_filter that overlap with binary
	guide_mask
	If keep_overlapping_objects is True, returns a mask containing only
	those objects that overlap guide_mask
	If keep_overlapping_objects is False, returns a mask containing all
	objects except the ones that overlap guide_mask
	'''
	# !!! NEEDS UNITTEST
	# Label each contiguous object in the image with a unique int
	# When identifying the objects, we must not count diagonal
	# neighbor pixels as belonging to the same object (this is VERY
	# important)
	# Default for cv2.connectedComponents is connectivity = 8
	label_num, labeled_mask = \
		cv2.connectedComponents(np.uint8(mask_to_filter), connectivity = 4)
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
	return(filtered_mask)


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