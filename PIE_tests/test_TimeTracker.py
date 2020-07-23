#!/usr/bin/python

import unittest
import cv2
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose
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

	@classmethod
	def setUpClass(self):
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
				[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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

class TestFindCentroidTransform(unittest.TestCase):
	'''
	Tests finding rigid-body affine transformation matrix that moves
	centroids from one timepoint to centroids from another timepoint
	'''

	@classmethod
	def setUpClass(self):
		self.time_tracker = \
			track_colonies._TimeTracker(timecourse_colony_prop_df)
		self.t1_data = self.time_tracker.timecourse_colony_prop_df[
			self.time_tracker.timecourse_colony_prop_df['timepoint'] == 1]

	def _make_affine_mat(self, angle_in_degrees, x_displacement, y_displacement):
		'''
		Creates affine transformation matrix to rotate image by
		angle_in_degrees and move it by x_displacement and
		y_displacement
		'''
		angle = angle_in_degrees*np.pi/180
		affine_mat = np.array([
			[np.cos(angle), -np.sin(angle), x_displacement],
			[np.sin(angle), np.cos(angle), y_displacement]])
		return(affine_mat)

	def _warp_centroids(self, timepoint_df, warp_mat):
		'''
		Warps positions of centroids in timepoint_df by warp_mat and
		returns dataframe with warped centroids
		'''
		centroids = np.float32(timepoint_df[['cX', 'cY']].to_numpy())
		warped_centroids = \
			np.squeeze(np.float32(cv2.transform(centroids[np.newaxis],
				warp_mat)))
		warped_df = timepoint_df.copy()
		warped_df[['cX', 'cY']] = warped_centroids
		return(warped_df)

	def test_simple_coord_transform(self):
		'''
		test whether correct matrix is found when self.t1_data is
		rotated and shifted
		'''
		expected_warp_mat = self._make_affine_mat(15, 40, -25)
			# this is a pretty big displacement
		warped_t1_data = self._warp_centroids(self.t1_data, expected_warp_mat)
		test_warp_mat = self.time_tracker._find_centroid_transform(self.t1_data,
			warped_t1_data)
		assert_allclose(expected_warp_mat, test_warp_mat, rtol = 1e-4)

	def test_coord_transform_missing_data(self):
		'''
		test whether correct matrix is found when self.t1_data is
		rotated and shifted, and rows are missing from both original and
		warped matrix
		'''
		expected_warp_mat = self._make_affine_mat(15, 40, -25)
			# this is a pretty big displacement
		warped_t1_data = self._warp_centroids(self.t1_data, expected_warp_mat)
		test_warp_mat = self.time_tracker._find_centroid_transform(
			self.t1_data.drop([2,3,11]), warped_t1_data.drop([5,8,9]))
		assert_allclose(expected_warp_mat, test_warp_mat, rtol = 1e-4)

	def test_coord_transform_missing_data_and_outlier(self):
		'''
		test whether correct matrix is found when self.t1_data is
		rotated and shifted, rows are missing from both original and
		warped matrix, and one of the datapoints is changed to be an
		outlier
		'''
		expected_warp_mat = self._make_affine_mat(15, 40, -25)
			# this is a pretty big displacement
		warped_t1_data = self._warp_centroids(self.t1_data, expected_warp_mat)
		warped_t1_data.loc[warped_t1_data.index[4], ['cX', 'cY']] = \
			warped_t1_data.loc[warped_t1_data.index[4], ['cX', 'cY']] + \
				np.array([200, 350])
		test_warp_mat = self.time_tracker._find_centroid_transform(
			self.t1_data.drop([2,3,11]), warped_t1_data.drop([5,8,9]))
		assert_allclose(expected_warp_mat, test_warp_mat, rtol = 1e-4)

	def test_too_few_points(self):
		'''
		test whether non-warp matrix is found when self.t1_data is
		too short to produce affine matrix
		'''
		expected_warp_mat = self._make_affine_mat(0, 0, 0)
		warped_t1_data = self.t1_data.iloc[0:2]
		test_warp_mat = self.time_tracker._find_centroid_transform(
			warped_t1_data, warped_t1_data)
		assert_allclose(expected_warp_mat, test_warp_mat, rtol = 1e-4)


