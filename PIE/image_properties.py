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
from PIE.image_coloring import create_color_overlay

class ImageAnalyzer(object):
	'''
	Runs image analysis and creates necessary output files
	'''
	# TODO: save all jpegs as jpeg2000s instead?
	def __init__(self, original_im, image_name, output_path, image_type = 'bright',
				hole_fill_area = np.inf, cleanup = False,
				max_proportion_exposed_edge = 0.75, cell_intensity_num = 1,
				save_extra_info = False, threshold_plot_width = 6,
				threshold_plot_height = 4, threshold_plot_dpi = 200,
				threshold_plot_filetype = 'png',
				create_pie_overlay = False, write_col_props_file = True, max_col_num = 1000):
		# TODO: Reorganize this section to make more sense; add more info about inputs
		# parameter for saving jpegs (>95 not recommended by PIL)
		self._jpeg_quality = 95
		# parameters for saving plots
		self._threshold_plot_width = threshold_plot_width
		self._threshold_plot_height = threshold_plot_height
		self._threshold_plot_dpi = threshold_plot_dpi
		self._threshold_plot_filetype = threshold_plot_filetype
		self._save_extra_info = save_extra_info
		self.original_im = original_im
		if image_type.lower() in ['bright','dark']:
			self.image_type = image_type.lower()
		else:
			raise ValueError(
				"image_type must be either 'bright' or 'dark'")
		self.hole_fill_area = hole_fill_area
		self.cleanup = cleanup
		self._create_pie_overlay = create_pie_overlay
		self.write_col_props_file = write_col_props_file
		self.max_proportion_exposed_edge = max_proportion_exposed_edge
		self.cell_intensity_num = cell_intensity_num
		self.max_col_num = max_col_num
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
			try:
				os.makedirs(current_path)
			except:
				pass

	def _prep_image(self):
		'''
		Reads the image, performs normalization
		If image_type is 'bright', reads as is
		If image_type is 'dark', inverts image before analysis
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
				norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
				# decrease bitdepth of norm_im to 8-bit while normalizing
		self.norm_im_8_bit = cv2.normalize(self.original_im, None, alpha=0, beta=(2**8-1),
			norm_type=cv2.NORM_MINMAX)
		if self.image_type == 'bright':
			self.edge_input_im = self.norm_im
		elif self.image_type == 'dark':
			self.edge_input_im = cv2.bitwise_not(self.norm_im)
		
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
					'threshold_info_'+self.image_name+'.csv')
		threshold_info = pd.DataFrame({
			'method':[threshold_method_name],
			'threshold':[threshold]
			}, index = [self.image_name])
		threshold_info.to_csv(threshold_info_file)

	def _find_cell_centers(self):
		'''
		Find cell centers and save threshold plot if necessary
		'''
		self.cell_centers, threshold_method_name, threshold_plot, \
			threshold, default_threshold_method_usage = \
			adaptive_threshold.threshold_image(
				self.norm_im, self.image_type, self.cell_intensity_num,
				self._save_extra_info
				)
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
		self.pie_edge_detector = \
			colony_edge_detect.EdgeDetector(
				self.edge_input_im, self.cell_centers, self.hole_fill_area,
				self.cleanup, self.max_proportion_exposed_edge)
		self.colony_mask = self.pie_edge_detector.run_edge_detection()

	def _save_jpeg(self):
		'''
		Writes jpeg of normalized input image
		'''
		# !!! NEEDS UNITTEST
		jpeg_path = os.path.join(self._image_output_dir_dict['jpgGRimages'],
			self.image_name + '.jpg')
		cv2.imwrite(jpeg_path, self.norm_im_8_bit,
			[cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])

	def _save_colony_mask(self):
		'''
		Saves labeled colony mask as tif file
		'''
		# !!! NEEDS UNITTEST
		colony_mask_path = \
			os.path.join(
				self._image_output_dir_dict['colony_masks'],
				self.image_name + '.tif'
				)
		cv2.imwrite(colony_mask_path, np.uint16(self.colony_property_finder.labeled_mask))

	def _save_boundary_im(self):
		'''
		Creates and saves an image in which the normalized image is
		overlaid with a boundary (contour) of the colony mask
		'''
		colony_boundary_mask = ported_matlab.bwperim(self.colony_mask)
		self.boundary_im = create_color_overlay(self.norm_im_8_bit,
			colony_boundary_mask, self._boundary_color, self._boundary_alpha,
			bitdepth = 8)
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
			self.cell_centers, self._center_color, self._cell_center_alpha,
			bitdepth = 8)
		colony_center_overlay_path = \
			os.path.join(self._image_output_dir_dict['colony_center_overlays'],
				self.image_name + '.jpg')
		cv2.imwrite(colony_center_overlay_path, self.colony_center_overlay,
			[cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])

	def _save_colony_properties(self):
		'''
		Measures and, if needed, records area and centroid in every
		detected colony
		'''
		# !!! NEEDS UNITTEST !!!
		self.colony_property_finder = \
			_ColonyPropertyFinder(self.colony_mask, self.max_col_num)
		self.colony_property_finder.measure_and_record_colony_properties()
		if self.write_col_props_file:
			colony_property_path = \
				os.path.join(
					self._image_output_dir_dict['single_im_colony_properties'],
					self.image_name + '.csv')
			colony_property_df_to_save = self.get_colony_property_df_to_save()
			colony_property_df_to_save.to_csv(colony_property_path,
				index = False)

	def _save_post_edge_detection_files(self):
		'''
		Saves colony mask, jpeg, regionprops, and if necessary, boundary_ims, etc
		'''
		# write input image as a jpeg
		self._save_jpeg()
		# measure and save colony properties
		self._save_colony_properties()
		# write colony mask as 16-bit tif file
		self._save_colony_mask()
		# if self._save_extra_info is True, save additional files
		if self._save_extra_info:
			self._save_boundary_im()
			self._save_colony_center_overlay()

	def get_colony_property_df_to_save(self):
		'''
		Returns csv-file-friendly self.colony_property_df (i.e. without
		space-consuming pixel_idx_list, Eroded_Colony_Mask,
		Eroded_Background_Mask columns)
		'''
		columns_to_save = \
			self.colony_property_finder.property_df.columns.difference(
				['pixel_idx_list', 'Eroded_Colony_Mask', 'Eroded_Background_Mask'])
		colony_property_df_to_save = \
			self.colony_property_finder.property_df[columns_to_save].copy()
		return(colony_property_df_to_save)

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
	def __init__(self, colony_mask, max_col_num, fluor_measure_expansion_pixels = 5):
		self.colony_mask = colony_mask
		self.max_col_num = max_col_num
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
		Finds connected components in an image (i.e. colonies)

		If number of colonies is higher than self.max_col_num, treats 
		as blank image

		Note that although connectivity = 4 is used at other steps, 
		connectivity is set to 8 for the colony identification step 
		(this is consistent with previous versions of PIE)
		'''
		[self._label_num, self.labeled_mask, self._stat_matrix,
			self._centroids] = \
			cv2.connectedComponentsWithStats(np.uint8(self.colony_mask),
				True, True, True, 8, cv2.CV_32S)
		# if more labels (besides background) than self.max_col_num,
		# recalculate properties from blank mask
		if self._label_num > (self.max_col_num+1):
			blank_mask = np.zeros_like(np.uint8(self.colony_mask))
			[self._label_num, self.labeled_mask, self._stat_matrix,
				self._centroids] = \
				cv2.connectedComponentsWithStats(
					blank_mask,
					True, True, True, 8, cv2.CV_32S)

	def _find_labels(self):
		'''
		Records the label of each colony as a string
		'''
		# start labels at 1, since '0' label should be background
		# (this assumption made elsewhere as well)
		labels = np.arange(1, self._label_num).astype(str)
		# remove background
		self.property_df['label'] = labels

	def _find_areas(self):
		'''
		Records the areas of each colony
		'''
		areas = self._stat_matrix[:,4]
		# remove background
		self.property_df['area'] = areas[1:]

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

	def _find_contour_props(self, single_colony_mask):
		'''
		Finds external perimeter and major axis length of best-fit
		ellipse of mask that should contain single colony
		NB: perimeter calculated here between centers of consecutive
		contour pixels; ellipse fitting for major axis length also
		happens differently than in matlab, resulting in slightly
		smaller values calculated by cv2
		'''
		colony_cont = \
			cv2.findContours(np.uint8(single_colony_mask), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_NONE)[0][0]
		colony_perim = cv2.arcLength(colony_cont, True)
		# documentation of fitEllipse seems to give minor and major axis
		# in reverse order, return max to be sure
		# fitEllipse doesn't work with <5 pixels, and produces erratic
		# results when there are 5 pixels in the object
		if np.sum(single_colony_mask) > 5:
			_, (axis_0_length, axis_1_length), _ = cv2.fitEllipse(colony_cont)
		else:
			# fitEllipse won't work, find max number of non-overlapping
			# pixels along length and width axes
			axis_0_length = np.sum(np.any(single_colony_mask, axis = 0))
			axis_1_length = np.sum(np.any(single_colony_mask, axis = 1))
		maj_axis_length = max([axis_0_length, axis_1_length])
		return(colony_perim, maj_axis_length)

	def _find_colonywise_properties(self):
		'''
		Loops through detected colonies and measures the properties that
		need to be measured one at a time
		'''
		self.property_df['perimeter'] = np.nan
		self.property_df['pixel_idx_list'] = None
		# don't loop through background
		for colony in range(1, self._label_num):
			current_colony_mask = self.labeled_mask == colony
			self.property_df.at[colony-1, 'perimeter'], \
				self.property_df.at[colony-1, 'major_axis_length'] = \
					self._find_contour_props(current_colony_mask)
			self.property_df.at[colony-1, 'pixel_idx_list'] = \
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

		If fluor_threshold is an empty string, defaults to np.inf
		'''
		if fluor_threshold == '' or \
			fluor_threshold is None or \
			np.isnan(fluor_threshold):
			fluor_threshold = np.inf
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
		All fluor properties must end in _flprop
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
			fluor_prop_dict['back_mean_ppix_flprop'] = np.mean(background_intensities)
			fluor_prop_dict['back_med_ppix_flprop'] = np.median(background_intensities)
			# calculate the variance of pixel intensities
			fluor_prop_dict['back_var_ppix_flprop'] = np.var(background_intensities)
		else:
			fluor_prop_dict['back_mean_ppix_flprop'] = np.nan
			fluor_prop_dict['back_med_ppix_flprop'] = np.nan
			fluor_prop_dict['back_var_ppix_flprop'] = np.nan
		if len(colony_intensities) > 0:
			# get the per pixel intensity for the colony excluding saturated
			# region at the colony edge
			fluor_prop_dict['col_mean_ppix_flprop'] = np.mean(colony_intensities)
			fluor_prop_dict['col_med_ppix_flprop'] = np.median(colony_intensities)
			# calculate the variance of pixel intensities
			fluor_prop_dict['col_var_ppix_flprop'] = np.var(colony_intensities)
			# calculate the upper quartile value of colony intensities
			fluor_prop_dict['col_upquartile_ppix_flprop'] = \
				np.quantile(colony_intensities, 0.75)
		else:
			fluor_prop_dict['col_mean_ppix_flprop'] = np.nan
			fluor_prop_dict['col_med_ppix_flprop'] = np.nan
			fluor_prop_dict['col_var_ppix_flprop'] = np.nan
			fluor_prop_dict['col_upquartile_ppix_flprop'] = np.nan
		return(fluor_prop_dict)

	def measure_and_record_colony_properties(self):
		'''
		Measures and records all colony properties
		'''
		self._find_connected_components()
		self._find_labels()
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
			colony_pixel_idx_list = row['pixel_idx_list']
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

def analyze_single_image(
	input_im_path,
	output_path,
	image_type,
	hole_fill_area = 'inf',
	cleanup = False,
	max_proportion_exposed_edge = 0.75,
	cell_intensity_num = 1,
	save_extra_info = True):
	'''
	Reads image from input_im_path and runs PIE colony detection on it,
	saving required files to output_path

	Does not cap number of colonies in image
	'''
	# analyzes all colonies in image
	max_col_num = np.inf
	image = cv2.imread(input_im_path, cv2.IMREAD_ANYDEPTH)
	image_name = os.path.splitext(os.path.basename(input_im_path))[0]
	image_analyzer = ImageAnalyzer(image, image_name, output_path,
		image_type, float(hole_fill_area), cleanup, max_proportion_exposed_edge,
		cell_intensity_num, save_extra_info, max_col_num = max_col_num)
	colony_mask, colony_property_df = image_analyzer.process_image()
	return(colony_mask, colony_property_df)
