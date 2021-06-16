#!/usr/bin/python

'''
Functions that are generally useful across testing code
'''

import cv2
import numpy as np
import os
from PIE.image_properties import create_color_overlay
from PIE.ported_matlab import bwperim


def show_mask_diffs(green_mask, mag_mask, im, im_name):
	'''
	Displays im with pixels unique to green_mask as green and pixels
	unique to mag_mask as magenta
	'''
	norm_im = \
		cv2.normalize(im, None, alpha=0, beta=(2**8-1),
			norm_type=cv2.NORM_MINMAX)
	green_bounds = bwperim(green_mask)
	mag_bounds = bwperim(mag_mask)
	green_pixels = np.copy(green_mask)
	green_pixels[mag_mask] = False
	mag_pixels = np.copy(mag_mask)
	mag_pixels[green_mask] = False
	green_im = create_color_overlay(norm_im, green_pixels, [0, 255, 0], 1)
	green_mag_im = create_color_overlay(green_im, mag_pixels, [255, 0, 255], 1)
	bound_im_1 = create_color_overlay(green_mag_im, green_bounds, [0, 255, 0], 0.5)
	bound_im_2 = create_color_overlay(bound_im_1, mag_bounds, [255, 0, 255], 0.5)
	out_path = os.path.join('tests', 'test_ims',
		('test_overlap_' + im_name + '.tif'))
	cv2.imwrite(out_path, bound_im_2 * 2**8)
