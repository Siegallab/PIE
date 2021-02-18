#!/usr/bin/python

'''
Colors images according to masks or keys
'''

import cv2
import numpy as np
import pandas as pd

def colorize_im(input_im, rgb_tuple):
	'''
	Colorizes grayscale image input_im so that pixel colors go from
	black to color specified by rgb_tuple

	input_im is an 8-bit cv2 image

	rgb_tuple is in the format (R,G,B), with a max value of 255
	'''
	color_image = np.copy(input_im)
	if len(np.shape(color_image)) == 2:
		color_image = cv2.cvtColor(np.float32(color_image), cv2.COLOR_GRAY2RGB)
	colorized_image = np.uint8(color_image*np.array(rgb_tuple)/255)
	return(colorized_image)

def overlay_color_im(color_im, color_mask, mask_alpha):
	'''
	Overlays color_mask over 3-D color_im with color_mask transparency 
	mask_alpha

	Only apply transparency in masked areas
	'''
	# apply transparency anywhere where color_mask has a color in at 
	# least one channel
	mask_bool = np.max(color_mask,axis=2)>0
	overlay_im = color_im.copy()
	overlay_im[mask_bool] = np.round(
		color_im[mask_bool].astype(float)*(1-mask_alpha) +
		color_mask[mask_bool].astype(float)*mask_alpha
		).astype(int)
	return(overlay_im)

def create_color_overlay(image, mask, mask_color, mask_alpha,
	bitdepth = None):
	'''
	Creates an rbg image of image with mask overlaid on it in
	mask_color with mask_alpha

	mask_color is a list of r, g, b values on a scale of 0 to 255

	If bitdepth is None, uses the max of supplied image as the max
	intensity
	'''
	# convert image to colot if necessary
	color_image = np.copy(image)
	if len(np.shape(color_image)) == 2:
		color_image = cv2.cvtColor(np.float32(color_image), cv2.COLOR_GRAY2RGB)
	# set max image intensity
	if bitdepth is None:
		max_intensity = np.max(color_image)
	else:
		max_intensity = 2**bitdepth-1
	# adjust mask_color by alpha, inverse, scale to image bitdepth
	mask_color_adjusted_tuple = \
		tuple(float(k)/255*max_intensity*mask_alpha for k in mask_color[::-1])
	color_image[mask] = \
		np.round(color_image[mask].astype(float) * (1-mask_alpha) +
			mask_color_adjusted_tuple).astype(int)
	return(color_image.astype(int))

def paint_by_numbers(labeled_im, color_key, background_rgb=(0,0,0)):
	'''
	Converts 2-D array labeled_im to 3-D RGB array based on color_key

	labels in labeled_im should serve as keys in color_key; values in 
	color_key should be (R,G,B) tuples

	background_rgb is the color to paint pixels not in color_key, as 
	an (R,G,B) tuple
	'''
	# create df of color keys (colors as columns, labels as rows)
	color_key_df_part = pd.DataFrame(
		color_key, index = ['R','G','B']
		).transpose()
	# create df of color keys (with background_rgb) for labels not 
	# listed in color_key
	label_list, label_idx = np.unique(labeled_im, return_inverse = True)
	background_labels = list(set(label_list)-set(color_key_df_part.index))
	background_key_df = pd.DataFrame(
		[background_rgb]*len(background_labels),
		index = background_labels,
		columns = ['R','G','B'])
	color_key_df = pd.concat([color_key_df_part, background_key_df])
	# reorder to match label_list order (and thus be compatible with
	# label_idx)
	color_key_df = color_key_df.loc[label_list]
	# use index to create images in every color
	color_im_list = []
	for color in ['R','G','B']:
		flat_im = color_key_df[color].to_numpy()[label_idx]
		current_im = flat_im.reshape(labeled_im.shape)
		color_im_list.append(current_im)
	colored_im = np.dstack(color_im_list)
	return(colored_im)

def get_boundary(bool_mask, bound_width):
	'''
	Returns image with boundaries of thickness bound_width for all 
	True objects in bool_mask; boundaries start inside bounds of 
	original objects

	bool_mask is a 2-D boolean np array

	bound_width is an integer
	'''
	mask_im = np.uint8(bool_mask)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	eroded_mask_im = cv2.erode(mask_im,kernel,iterations=bound_width)
	bound_mask_im = mask_im-eroded_mask_im
	bound_mask_bool = bound_mask_im.astype(bool)
	return(bound_mask_bool)

def safe_uint8_convert(im):
	'''
	Converts im to uint8, but sets oversaturated pixels to max 
	intensity and throws oversaturation warning
	'''
	input_im = im.copy()
	oversat_bool = input_im > 255
	if np.sum(oversat_bool) > 0:
		with warnings.catch_warnings():
			warnings.simplefilter("always")
			warnings.warn('Some pixels are oversaturated', UserWarning)
	input_im[oversat_bool] = 255
	input_im = np.uint8(input_im)
	return(input_im)
