#!/usr/bin/python

import unittest
import os
import shutil
import warnings
import numpy as np
from numpy.testing import assert_array_equal
from PIE.image_properties import _ImageAnalyzer

class TestSetUpFolders(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.output_path = 'PIE_tests/test_output_dir'

	def setUp(self):
		self.image_analyzer_standin = object.__new__(_ImageAnalyzer)

	def test_folder_creation(self):
		'''
		Tests that PIE quadrants correctly creates folders in target directory
		'''
		self.image_analyzer_standin._set_up_folders(self.output_path)
		self.assertTrue(os.path.exists('PIE_tests/test_output_dir/jpgGRimages'))
		self.assertTrue(os.path.exists('PIE_tests/test_output_dir/boundary_ims'))
		self.assertTrue(os.path.exists('PIE_tests/test_output_dir/colony_masks'))
		self.assertTrue(os.path.exists('PIE_tests/test_output_dir/threshold_plots'))
		self.assertTrue(os.path.exists('PIE_tests/test_output_dir/colony_center_overlays'))
		self.assertTrue(os.path.exists('PIE_tests/test_output_dir/single_im_colony_properties'))

	def test_folder_creation(self):
		'''
		Tests that PIE quadrants correctly creates folders in target
		directory without throwing a warning even when one of the
		folders already exists
		'''
		os.makedirs('PIE_tests/test_output_dir/jpgGRimages')
		with warnings.catch_warnings(record=True) as w:
			# Cause all warnings to always be triggered.
			warnings.simplefilter("always")
			self.image_analyzer_standin._set_up_folders(self.output_path)
			# Check that 2 warnings issued
			self.assertEqual(len(w),0)
		self.assertTrue(os.path.exists('PIE_tests/test_output_dir/jpgGRimages'))
		self.assertTrue(os.path.exists('PIE_tests/test_output_dir/boundary_ims'))
		self.assertTrue(os.path.exists('PIE_tests/test_output_dir/colony_masks'))
		self.assertTrue(os.path.exists('PIE_tests/test_output_dir/threshold_plots'))
		self.assertTrue(os.path.exists('PIE_tests/test_output_dir/colony_center_overlays'))
		self.assertTrue(os.path.exists('PIE_tests/test_output_dir/single_im_colony_properties'))

	def tearDown(self):
		# remove directories created
		shutil.rmtree(self.output_path)

class TestPrepImage(unittest.TestCase):

	def setUp(self):
		self.image_analyzer_standin = object.__new__(_ImageAnalyzer)
		self.image_analyzer_standin.original_im = \
			np.uint16([[0, 1024, 2047], [0, 2047, 0]])

	def test_bright(self):
		'''
		Tests input_im and norm_im creation for brightfield 11-bit image
		'''
		self.image_analyzer_standin.image_type = 'bright'
		self.image_analyzer_standin._prep_image()
		expected_norm_im = np.uint16([[0, 32784, 65535], [0, 65535, 0]])
		expected_input_im = np.copy(expected_norm_im)
		assert_array_equal(expected_norm_im,
			self.image_analyzer_standin.norm_im)
		assert_array_equal(expected_input_im,
			self.image_analyzer_standin.input_im)

	def test_dark(self):
		'''
		Tests input_im and norm_im creation for phase constrast 11-bit
		image, for which image should be inverted
		'''
		self.image_analyzer_standin.image_type = 'dark'
		self.image_analyzer_standin._prep_image()
		expected_norm_im = np.uint16([[0, 32784, 65535], [0, 65535, 0]])
		expected_input_im = np.uint16([[65535, 32751, 0], [65535, 0, 65535]])
		assert_array_equal(expected_norm_im,
			self.image_analyzer_standin.norm_im)
		assert_array_equal(expected_input_im,
			self.image_analyzer_standin.input_im)


if __name__ == '__main__':
	unittest.main()