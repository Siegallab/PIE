#!/usr/bin/python

import unittest
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
from PIE import track_colonies

# load in a test timecourse colony property dataframe
# NB: this case is quite pathological, with poor tracking, but this is
# probably better for testing
timecourse_colony_prop_df = \
	pd.read_csv(os.path.join('PIE_tests','test_ims',
		'SL_170619_2_GR_small_xy0001_phase_colony_data_tracked.csv'),
	index_col = 0)

class TestGetOverlap(unittest.TestCase):
	'''
	Tests getting overlap of colonies between current and next timepoint
	'''

	def setUp(self):
		self.time_tracker = \
			track_colonies._TimeTracker(timecourse_colony_prop_df)

	def test_get_overlap_t5t6(self):
		'''
		Tests finding overlap between timepoints 5 and 6 of
		timecourse_colony_prop_df
		Checked against results of previous matlab code (and manual
		confirmation of most rows)
		'''
		curr_timepoint = 5
		next_timepoint = 6
		expected_overlap_df = pd.DataFrame(
			np.array([
				[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
				dtype = bool),
			index = np.arange(0,14), columns = np.arange(0,16))
		test_overlap_df = \
			self.time_tracker._get_overlap(curr_timepoint, next_timepoint)
		assert_frame_equal(expected_overlap_df, test_overlap_df)