#!/usr/bin/python

import unittest
import os
import numpy as np
import pandas as pd
import shutil
#import pandas.testing.assert_frame_equal
#import numpy.testing.assert_equal
from PIE.analysis_configuration import AnalysisConfig, _IndivImageRetriever

class TestAnalysisConfigInit(unittest.TestCase):
	'''
	Tests initialization of AnalysisConfig object
	'''

	def setUp(self):
		self.phase_num = 1
		self.hole_fill_area = np.inf
		self.cleanup = True
		self.perform_registration = True
		self.max_proportion_exposed_edge = 0.75
		self.cell_intensity_num = 2
		self.input_path = os.path.join('PIE_test_data','IN','SL_170619_2_GR_small')
		self.output_path = os.path.join('tests','temp_out')
		self.im_file_extension = 'tif'
		self.label_order_list = ['channel', 'timepoint', 'position']
		self.max_xy_position_num = 1000
		self.first_timepoint = 1
		self.max_timepoint_num = 10
		self.timepoint_spacing = 3600
		self.timepoint_label_prefix = 't'
		self.position_label_prefix = 'xy'
		self.main_channel_label = ''
		self.main_channel_imagetype = 'bright'
		self.fluor_channel_df = pd.DataFrame(columns = ['fluor_channel_label',
			'fluor_channel_column_name', 'fluor_threshold'])
		self.im_format = 'individual'
		self.extended_display_positions = [1, 4, 11]
		self.xy_position_vector = range(1,1001)
		self.combined_gr_write_path = \
			os.path.join(self.output_path,'growth_rates_combined.csv')
		self.combined_tracked_properties_write_path = \
			os.path.join(self.output_path,'colony_properties_combined.csv')
		self.col_properties_output_folder = \
			os.path.join(self.output_path,'positionwise_colony_properties')
		self.movie_folder = \
			os.path.join(self.output_path,'movies')
		self.phase_output_path = os.path.join(self.output_path,'phase_1')
		self.phase_col_property_mats_output_folder = \
			os.path.join(self.phase_output_path,
				'positionwise_colony_property_matrices')
		self.phase_gr_write_path = \
			os.path.join(self.phase_output_path,'growth_rates.csv')
		self.filtered_colony_file = \
			os.path.join(self.phase_output_path,'filtered_colonies.csv')
		self.minimum_growth_time = 4
		self.growth_window_timepoints = 7
		self.max_area_pixel_decrease = 500.0
		self.max_area_fold_decrease = 2.0
		self.max_area_fold_increase = 6.0
		self.min_colony_area = 30.0
		self.max_colony_area = np.inf
		self.min_correlation = 0.9
		self.min_foldX = 0.0
		self.min_neighbor_dist = 100.0
		self.max_colony_num = 1000

	def test_init(self):

		expected_analysis_config = \
			object.__new__(AnalysisConfig)
		expected_analysis_config.phase_num = self.phase_num
		expected_analysis_config.max_timepoint_num = self.max_timepoint_num
		expected_analysis_config.max_xy_position_num = self.max_xy_position_num
		expected_analysis_config.hole_fill_area = self.hole_fill_area
		expected_analysis_config.cleanup = self.cleanup
		expected_analysis_config.perform_registration = self.perform_registration
		expected_analysis_config.max_proportion_exposed_edge = \
			self.max_proportion_exposed_edge
		expected_analysis_config.cell_intensity_num = self.cell_intensity_num
		expected_analysis_config.minimum_growth_time = self.minimum_growth_time
		expected_analysis_config.growth_window_timepoints = \
			self.growth_window_timepoints
		expected_analysis_config.max_area_pixel_decrease = \
			self.max_area_pixel_decrease
		expected_analysis_config.max_area_fold_decrease = \
			self.max_area_fold_decrease
		expected_analysis_config.max_area_fold_increase = \
			self.max_area_fold_increase
		expected_analysis_config.min_colony_area = self.min_colony_area
		expected_analysis_config.max_colony_area = self.max_colony_area
		expected_analysis_config.min_correlation = self.min_correlation
		expected_analysis_config.min_foldX = self.min_foldX
		expected_analysis_config.min_neighbor_dist = self.min_neighbor_dist
		expected_analysis_config.max_colony_num = self.max_colony_num
		expected_analysis_config.input_path = self.input_path
		expected_analysis_config.output_path = self.output_path
		expected_analysis_config.im_file_extension = self.im_file_extension
		expected_analysis_config.label_order_list = self.label_order_list
			# TODO: test supplying incomplete list of labels
		expected_analysis_config.phase_output_path = self.phase_output_path
		expected_analysis_config.phase_col_property_mats_output_folder = \
			self.phase_col_property_mats_output_folder
		expected_analysis_config.phase_gr_write_path = self.phase_gr_write_path
		expected_analysis_config.filtered_colony_file = \
			self.filtered_colony_file
		expected_analysis_config.combined_gr_write_path = \
			self.combined_gr_write_path
		expected_analysis_config.combined_tracked_properties_write_path = \
			self.combined_tracked_properties_write_path
		expected_analysis_config.col_properties_output_folder = \
			self.col_properties_output_folder
		expected_analysis_config.movie_folder = self.movie_folder
		expected_analysis_config.timepoint_list = range(1,11)
		expected_analysis_config.timepoint_dict = \
			dict(zip(range(1,11), np.arange(1,11)*3600.0))
		expected_analysis_config.xy_position_vector = self.xy_position_vector
		expected_analysis_config.timepoint_label_prefix = \
			self.timepoint_label_prefix
		expected_analysis_config.position_label_prefix = \
			self.position_label_prefix
		expected_analysis_config.first_timepoint_time = 3600.0
		expected_analysis_config.size_ref_im = 't01xy0001.tif'
		expected_analysis_config.im_width = 850
		expected_analysis_config.im_height = 720
		expected_analysis_config.main_channel_imagetype = \
			self.main_channel_imagetype
		expected_analysis_config.main_channel_label = self.main_channel_label
		expected_analysis_config.fluor_channel_df = self.fluor_channel_df
		expected_analysis_config.image_retriever = _IndivImageRetriever()
		expected_analysis_config.extended_display_positions = [1,4,11]
		# create analysis config
		test_analysis_config = AnalysisConfig(
			self.phase_num, self.hole_fill_area, self.cleanup, 
			self.perform_registration, self.max_proportion_exposed_edge, 
			self.cell_intensity_num, self.input_path, self.output_path,
			self.im_file_extension, self.label_order_list,
			self.max_xy_position_num, self.first_timepoint,
			self.max_timepoint_num, self.timepoint_spacing,
			self.timepoint_label_prefix, self.position_label_prefix,
			self.main_channel_label, self.main_channel_imagetype,
			self.fluor_channel_df, self.im_format,
			self.extended_display_positions, self.xy_position_vector,
			self.minimum_growth_time,
			self.growth_window_timepoints, self.max_area_pixel_decrease,
			self.max_area_fold_decrease, self.max_area_fold_increase,
			self.min_colony_area, self.max_colony_area, self.min_correlation,
			self.min_foldX, self.min_neighbor_dist, self.max_colony_num
			)
		# create dicts to compare objects
		test_attribute_dict = vars(test_analysis_config)
		expected_attribute_dict = vars(expected_analysis_config)
		# check special attributes
		self.assertEqual(type(expected_attribute_dict['image_retriever']),
			type(test_attribute_dict['image_retriever']))
		pd.testing.assert_frame_equal(
			expected_attribute_dict['fluor_channel_df'],
			test_attribute_dict['fluor_channel_df'])
		np.testing.assert_equal(
			expected_attribute_dict['main_channel_label'],
			test_attribute_dict['main_channel_label'])
		# remove special attributes from dicts
		special_attributes = \
			['image_retriever', 'fluor_channel_df', 'main_channel_label']
		[expected_attribute_dict.pop(k) for k in special_attributes]
		[test_attribute_dict.pop(k) for k in special_attributes]
		self.assertEqual(expected_attribute_dict, test_attribute_dict)
		# check for existance of phase-specific directories
		self.assertTrue(os.path.exists(self.phase_col_property_mats_output_folder))
		self.assertTrue(os.path.exists(self.phase_output_path))

	def tearDown(self):
		shutil.rmtree(self.col_properties_output_folder)
		shutil.rmtree(self.phase_output_path)
		shutil.rmtree(self.output_path)


if __name__ == '__main__':
	unittest.main()