#!/usr/bin/python

'''
Performs colony edge detection
'''

import cv2
import numpy as np

class _EdgeDetector(object):
	'''
	Detects edges in input_im using PIE algorithm
	'''

	def __init__(self, input_im_path):
		'''
		Read input_im_path as grayscale image of arbitrary bitdepth
		'''
		self.input_im = cv2.imread(input_im_path, cv2.IMREAD_ANYDEPTH)
		# create 'pie' quadrants
		self._get_pie_pieces()

	def _get_pie_pieces(self):
		'''
		By mutliplying x- by y- gradient directions, we get 4 sets if images,
		which, together, make up every pixel in the image; conveniently, each
		cell is represented as 4 separate quarters of a pie, each of which is
		surrounded by nothing.
		'''
		### ??? NEEDS UNITTEST ??? ###
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

	def _create_inital_overlay(self):
		'''
		Runs edge detection without cleanup
		'''
		### ??? NEEDS UNITTEST ??? ###
		# identify cell centers with adaptive thresholding of tophat im
		cell_centers = _ThresholdFinder.threshold_image(self.input_im)
		# identify 'pie pieces' that overlap cell centers, and combine
		# them into a composite overlay
		initial_overlay = np.zeros(self.input_im.shape, dtype = bool)
		for pie_quad_name, pie_piece_quadrant in \
			self.pie_piece_dict.iteritems():
			current_cell_pieces = \
				pie_piece_quadrant.id_center_overlapping_pie_pieces(
					cell_centers)
			initial_overay = np.logical_or(initial_overay, current_cell_pieces)
		return(initial_overay)


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
		# Label each contiguous object in the image with a unique int
		# When identifying the objects, we must not count diagonal
		# neighbor pixels as belonging to the same object (this is VERY
		# important)
		# Default for cv2.connectedComponents is connectivity = 8
		label_num, labeled_pie_mask = \
			cv2.connectedComponents(np.uint8(self.pie_mask), connectivity = 4)
		# Identify the positions in each labeled_objects that overlap
		# with cell_center_mask
		center_pie_overlap = labeled_pie_mask[cell_center_mask]
		# Identify the label numbers (allowed_objects) that overlap with
		# cell_center_mask
		allowed_labels_with_backgrnd = np.unique(center_pie_overlap)
		# excluding the background (where labeled_objects==0) is key!
		allowed_labels = \
			allowed_labels_with_backgrnd[allowed_labels_with_backgrnd != 0]
		# identify pixels in labeled_pie_mask whose labels are in
		# allowed_labels
		cell_overlapping_pie_mask = \
			np.isin(labeled_pie_mask, allowed_labels)
		return(cell_overlapping_pie_mask)




class _ThresholdFinder(object):
	'''
	Finds adaptive threshold for image
	'''

	def threshold_image(self, img):
		pass


if __name__ == '__main__':
	input_im = cv2.imread('/Users/plavskin/Documents/yeast stuff/pie_paper_v2/plotcode/Fig2-5_ims/input_ims/xy01_12ms_3702.tif',
		cv2.IMREAD_ANYDEPTH)
	input_im_2 = input_im*32
	print(np.amax(input_im))
	print(np.amax(input_im_2))
	print(input_im.shape)
	cv2.imshow('Input Image', input_im_2)
	cv2.waitKey(10000)
	cv2.destroyAllWindows()