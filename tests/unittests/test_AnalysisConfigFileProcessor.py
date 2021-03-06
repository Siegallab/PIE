#!/usr/bin/python

import unittest
import os
import numpy as np
import pandas as pd
#import pandas.testing.assert_frame_equal
#import numpy.testing.assert_equal
from PIE.analysis_configuration import _AnalysisConfigFileProcessor, \
	AnalysisConfig

class TestProcessAnalysisConfigFile(unittest.TestCase):

	def test_single_phase_input(self):
		'''
		Tests input with single growth rate phase
		'''
		analysis_config_file = 'sample_PIE_setup_files/gr_phase_setup.csv'
		analysis_config_file_processor = _AnalysisConfigFileProcessor()
		test_analysis_config_obj_df = \
			analysis_config_file_processor.process_analysis_config_file(
				analysis_config_file)
		expected_analysis_config = \
			AnalysisConfig(
				1, float('Inf'), False, True, 0.75, 1,
				'PIE_test_data/IN/SL_170619_2_GR_small', 'PIE_test_data/out/SL_170619_2_GR_small', 'tif',
				['timepoint', 'position'], 1000, 1, 10, 3600, 't',
				'xy', '','bright', pd.DataFrame(columns = ['fluor_channel_label',
					'fluor_channel_column_name', 'fluor_threshold', 'fluor_timepoint']), 'individual',
				[4, 11], range(1,1001), 4, 7, 500.0, 2.0, 6.0, 30, np.inf, 0.9, 0.0, 100.0, 1000)
		expected_analysis_config_obj_df = \
			pd.DataFrame({'analysis_config': [expected_analysis_config],
				'postphase_analysis_config': [None]}, index = [1])
		self.assertListEqual(expected_analysis_config_obj_df.index.tolist(),
			test_analysis_config_obj_df.index.tolist())
		self.assertListEqual(expected_analysis_config_obj_df.columns.tolist(),
			test_analysis_config_obj_df.columns.tolist())
		self.assertEqual(expected_analysis_config_obj_df.at[
			1, 'postphase_analysis_config'],
			test_analysis_config_obj_df.at[
				1, 'postphase_analysis_config'])
		test_analysis_config = \
			test_analysis_config_obj_df.at[1, 'analysis_config']
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

if __name__ == '__main__':
	unittest.main()