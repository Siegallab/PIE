#!/usr/bin/python

'''
Analyzes single-image colony properties
'''

import csv
import cv2
import numpy as np
import os
import pandas as pd
from PIE import adaptive_threshold, ported_matlab, colony_edge_detect

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
		self.colony_mask = colony_edge_detect.get_mask(self.input_im,
			self.cell_centers, self.hole_fill_area, self.cleanup,
			self.max_proportion_exposed_edge)

	def _save_jpeg(self):
		'''
		Writes jpeg of original input image
		'''
		# !!! NEEDS UNITTEST
		jpeg_path = os.path.join(self._image_output_dir_dict['jpgGRimages'],
			self.image_name + '.jpg')
		# check whether original_im is > 8-bit; if so, decrease bitdepth
		jpeg_im = np.copy(self.original_im)
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
		self.colony_property_finder = _ColonyPropertyFinder(self.colony_mask)
		self.colony_property_finder.measure_and_record_colony_properties()
		colony_property_path = \
			os.path.join(
				self._image_output_dir_dict['single_im_colony_properties'],
				self.image_name + '.csv')
		columns_to_save = \
			self.colony_property_finder.property_df.columns.difference(
				['PixelIdxList', 'Eroded_Colony_Mask', 'Eroded_Background_Mask'])
		colony_property_df_to_save = \
			self.colony_property_finder.property_df[columns_to_save]
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
		return(self.colony_mask, self.colony_property_finder.property_df)

	def set_up_fluor_measurements(self):
		'''
		Sets up for measuring fluorescence
		'''
		self.colony_property_finder.set_up_fluor_measurements()

	def measure_fluorescence(self, fluor_im, fluor_channel_name,
		fluor_threshold):
		'''
		Measures fluorescence in fluor_im
		'''
		self.colony_property_finder.make_fluor_measurements(fluor_im,
			fluor_channel_name, fluor_threshold)

class _ColonyPropertyFinder(object):
	'''
	Measures and holds properties of colonies based on a colony mask
	'''
	def __init__(self, colony_mask, fluor_measure_expansion_pixels = 5):
		self.colony_mask = colony_mask
		self.property_df = pd.DataFrame()
		# set up erosion mask for removing bright saturated region
		# around colony in fluorescence images
		self._fluor_mask_erosion_kernel = np.uint8([
			[0,0,1,0,0], [0,1,1,1,0], [1,1,1,1,1], [0,1,1,1,0], [0,0,1,0,0]])
			# !!! TODO: MAKE EROSION OPTIONAL!!!
		# set number of pixels that colony bounding box should be
		# expanded in each direction for calculation of background
		# fluorescence
		self._fluor_measure_expansion_pixels = fluor_measure_expansion_pixels

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
		# remove background bounding box
		self.property_df[['bb_x_left','bb_y_top','bb_width','bb_height']] = \
			pd.DataFrame(self._stat_matrix[1:, 0:4])

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

	def _pixel_idx_list_to_mask(self, pixel_idx_list, mask_shape):
		'''
		Creates a boolean nd array of shape mask_shape in which postions
		specified in pixel_idx_list are True and every other pixel is
		False
		'''
		mask = np.zeros(mask_shape, dtype = bool)
		mask.flat[np.fromstring(pixel_idx_list,	dtype = int, sep = ' ')] = True
		return(mask)

	def _expand_bounding_box(self, bounding_box_series, im_row_num, im_col_num,
		expansion_pixels):
		'''
		Takes in bounding_box_series, a pandas series containing
		bounding box info for a single colony
		Returns indices for first and last+1 pixels, in x and y
		direction, of bounding box expanded on each side by size
		expansion_pixels
		'''
		y_start = np.max(
			[0,
				np.int(np.ceil(bounding_box_series['bb_y_top']) -
					expansion_pixels)])
		x_start = np.max(
			[0,
				np.int(np.ceil(bounding_box_series['bb_x_left']) -
					expansion_pixels)])
		y_range_end = np.min(
			[im_row_num,
				np.int(np.ceil(bounding_box_series['bb_y_top'] + 
					bounding_box_series['bb_height']) +
					expansion_pixels)])
		x_range_end = np.min(
			[im_col_num,
				np.int(np.ceil(bounding_box_series['bb_x_left'] +
					bounding_box_series['bb_width']) +
					expansion_pixels)])
		return(y_start, x_start, y_range_end, x_range_end)

	def _subset_im_by_bounding_box(self, im, bounding_box_series,
		expansion_pixels):
		'''
		Returns the box in image im (a numpy array) defined by
		bounding_box_series
		'''
		bb_y_start, bb_x_start, bb_y_range_end, bb_x_range_end = \
			self._expand_bounding_box(bounding_box_series, im.shape[0],
				im.shape[1], expansion_pixels)
		bbox_im = im[bb_y_start:bb_y_range_end, bb_x_start:bb_x_range_end]
		return(bbox_im)

	def _get_eroded_bounded_mask(self, mask, colony_bounding_box_series,
		expansion_pixels):
		'''
		Returns the expanded bounding_box area of mask, eroded by
		self._fluor_mask_erosion_kernel
		'''
		# calculate expanded bounding box
		bbox_mask = \
			self._subset_im_by_bounding_box(mask, colony_bounding_box_series,
				expansion_pixels)
		bbox_mask_eroded = \
			cv2.erode(np.uint8(bbox_mask), self._fluor_mask_erosion_kernel,
				iterations = 1).astype(bool)
		return(bbox_mask_eroded)

	def _get_filtered_intensities(self, fluor_image, mask, fluor_threshold):
		'''
		Returns a 1D float numpy array of pixels in fluor_image that are
		True in mask and below fluor_threshold
		'''
		mask_intensities_unfiltered = fluor_image[mask]
		mask_intensities_filtered = \
			mask_intensities_unfiltered[
			mask_intensities_unfiltered <= fluor_threshold]
		return(mask_intensities_filtered.astype(float))

	def _measure_colony_fluor_properties(self, fluor_bbox_image,
		single_colony_bbox_mask_eroded, background_bbox_mask_eroded,
		fluor_threshold):
		'''
		Measures and records intensities of single colony and background
		masks in fluor_channel
		'''
		# create a dictionary to hold fluorescent properties
		fluor_prop_dict = dict()
		# get the fluorescence values within colony without the
		# saturated region at colony edge
		colony_intensities = \
			self._get_filtered_intensities(fluor_bbox_image,
				single_colony_bbox_mask_eroded, fluor_threshold)
		# get the fluorescence values in non-colony background within
		# the bounding box, without saturated regions around colony
		# edges
		background_intensities = \
			self._get_filtered_intensities(fluor_bbox_image,
				background_bbox_mask_eroded, fluor_threshold)
		if len(background_intensities) > 0:
			# get the per pixel intensity for the background near the colony
			fluor_prop_dict['back_mean_ppix'] = np.mean(background_intensities)
			fluor_prop_dict['back_med_ppix'] = np.median(background_intensities)
			# calculate the variance of pixel intensities
			fluor_prop_dict['back_var_ppix'] = np.var(background_intensities)
		else:
			fluor_prop_dict['back_mean_ppix'] = np.nan
			fluor_prop_dict['back_med_ppix'] = np.nan
			fluor_prop_dict['back_var_ppix'] = np.nan
		if len(colony_intensities) > 0:
			# get the per pixel intensity for the colony excluding saturated
			# region at the colony edge
			fluor_prop_dict['col_mean_ppix'] = np.mean(colony_intensities)
			fluor_prop_dict['col_med_ppix'] = np.median(colony_intensities)
			# calculate the variance of pixel intensities
			fluor_prop_dict['col_var_ppix'] = np.var(colony_intensities)
			# calculate the upper quartile value of colony intensities
			fluor_prop_dict['col_upquartile_ppix'] = \
				np.quantile(colony_intensities, 0.75)
		else:
			fluor_prop_dict['col_mean_ppix'] = np.nan
			fluor_prop_dict['col_med_ppix'] = np.nan
			fluor_prop_dict['col_var_ppix'] = np.nan
			fluor_prop_dict['col_upquartile_ppix'] = np.nan
		return(fluor_prop_dict)

	def measure_and_record_colony_properties(self):
		'''
		Measures and records all colony properties
		'''
		self._find_connected_components()
		self._find_areas()
		self._find_centroids()
		self._find_bounding_box()
		self._find_colonywise_properties()

	def set_up_fluor_measurements(self):
		'''
		Sets up for measuring fluorescence by calculating eroded colony
		and background masks for every colony
		'''
		self.property_df['Eroded_Colony_Mask'] = None
		self.property_df['Eroded_Background_Mask'] = None
		# create mask of background, excluding colonies
		background_mask = np.invert(self.colony_mask)
		for idx, row in self.property_df.iterrows():
			colony_pixel_idx_list = row['PixelIdxList']
			colony_bounding_box_series = \
				row[['bb_x_left','bb_y_top','bb_width','bb_height']]
			# create a mask of the current colony
			single_colony_mask = \
				self._pixel_idx_list_to_mask(colony_pixel_idx_list,
					self.colony_mask.shape)
			# get mask for colony without saturated region at colony edge
			self.property_df.at[idx, 'Eroded_Colony_Mask'] = \
				self._get_eroded_bounded_mask(single_colony_mask,
					colony_bounding_box_series,
					self._fluor_measure_expansion_pixels)
			# get mask for non-colony background within the bounding box
			# without saturated regions around colony edge
			self.property_df.at[idx, 'Eroded_Background_Mask'] = \
				self._get_eroded_bounded_mask(background_mask,
					colony_bounding_box_series,
					self._fluor_measure_expansion_pixels)

	def make_fluor_measurements(self, fluor_im, fluor_channel_name,
		fluor_threshold):
		'''
		Measures intensity of each colony in fluor_im
		'''
		### NEEDS UNITTEST
		for idx, row in self.property_df.iterrows():
			colony_bounding_box_series = \
				row[['bb_x_left','bb_y_top','bb_width','bb_height']]
			# create a subimage of fluor_image
			fluor_bbox_image = \
				self._subset_im_by_bounding_box(fluor_im,
					colony_bounding_box_series,
					self._fluor_measure_expansion_pixels)
			# retrieve current colony and background masks
			single_colony_bbox_mask_eroded = row['Eroded_Colony_Mask']
			background_bbox_mask_eroded = row['Eroded_Background_Mask']
			# get dictionary of colony fluorescence properties
			colony_fluor_prop_dict = \
				self._measure_colony_fluor_properties(fluor_bbox_image,
					single_colony_bbox_mask_eroded, background_bbox_mask_eroded,
					fluor_threshold)
			# add fluor_channel_name to names of fluorescence properties
			channel_fluor_prop_names = \
				['_'.join([pname, fluor_channel_name]) for pname in
					colony_fluor_prop_dict.keys()]
			# create columns in self.property_df if they don't already
			# exist
			for pname in channel_fluor_prop_names:
				if pname not in self.property_df.columns:
					self.property_df[pname] = np.nan
			# add colony fluorescent properties to current row
			self.property_df.at[idx, channel_fluor_prop_names] = \
				colony_fluor_prop_dict.values()

def create_color_overlay(image, mask, mask_color, mask_alpha):
	'''
	Creates an rbg image of image with mask overlaid on it in
	mask_color with mask_alpha
	mask_color is a list of r, g, b values on a scale of 0 to 255
	'''
	# TODO: Doesn't work when image has a max of 0 (because of np.max(color_image) issue)
	# check whether image is grayscale; if so, convert to rgb
	color_image = np.copy(image)
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
	colony_mask, colony_property_df = image_analyzer.process_image()
	return(colony_mask, colony_property_df)

def detect_image_colonies(image, image_name, output_path, image_type,
	hole_fill_area, cleanup, max_proportion_exposed_edge, save_extra_info):
	'''
	Runs PIE colony detections on image
	Returns colony mask and image property df
	'''
	image_analyzer = _ImageAnalyzer(image, image_name, output_path,
		image_type, hole_fill_area, cleanup, max_proportion_exposed_edge,
		save_extra_info)
	colony_mask, colony_property_df = image_analyzer.process_image()
	return(colony_mask, colony_property_df)

# !!! NB: explain in doc that this assumes that there will never be two images with the same name being placed in the same output folder
