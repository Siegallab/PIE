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
# NB: this case is quite pathological, preliminary analysis on bad
# images, with poor tracking, but this is probably better for testing
timecourse_colony_prop_df = \
	pd.read_csv(os.path.join('tests','test_ims',
		'SL_170619_2_GR_small_xy0001_phase_colony_data_tracked.csv'),
	index_col = 0)

satellite_prop_df = \
	pd.read_csv(os.path.join('tests','test_ims',
		'test_sat_data.csv'),
	index_col = 0)

class TestGetOverlap(unittest.TestCase):
	'''
	Tests getting overlap of colonies between current and next timepoint
	'''

	def setUp(self):
		self.colony_tracker = \
			track_colonies.ColonyTracker()

	def test_get_overlap_t5t6(self):
		'''
		Tests finding overlap between timepoints 5 and 6 of
		timecourse_colony_prop_df
		Checked against results of previous matlab code (and manual
		confirmation of most rows)
		'''
		tp_5_data = \
			timecourse_colony_prop_df[
				timecourse_colony_prop_df.timepoint == 5]
		# get colony properties at next timepoint
		tp_6_data = \
			timecourse_colony_prop_df[
				timecourse_colony_prop_df.timepoint == 6]
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
			index = tp_5_data.index, columns = tp_6_data.index)
		test_overlap_df = \
			self.colony_tracker._get_overlap(tp_5_data, tp_6_data)
		assert_frame_equal(expected_overlap_df, test_overlap_df)

class TestFindCentroidTransform(unittest.TestCase):
	'''
	Tests finding rigid-body affine transformation matrix that moves
	centroids from one timepoint to centroids from another timepoint
	'''

	@classmethod
	def setUpClass(self):
		self.colony_tracker = \
			track_colonies.ColonyTracker()
		self.t1_data = timecourse_colony_prop_df[
			timecourse_colony_prop_df['timepoint'] == 1]

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
		test_warp_mat = self.colony_tracker._find_centroid_transform(self.t1_data,
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
		test_warp_mat = self.colony_tracker._find_centroid_transform(
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
		test_warp_mat = self.colony_tracker._find_centroid_transform(
			self.t1_data.drop([2,3,11]), warped_t1_data.drop([5,8,9]))
		assert_allclose(expected_warp_mat, test_warp_mat, rtol = 1e-4)

	def test_too_few_points(self):
		'''
		test whether non-warp matrix is found when self.t1_data is
		too short to produce affine matrix
		'''
		expected_warp_mat = self._make_affine_mat(0, 0, 0)
		warped_t1_data = self.t1_data.iloc[0:2]
		test_warp_mat = self.colony_tracker._find_centroid_transform(
			warped_t1_data, warped_t1_data)
		assert_allclose(expected_warp_mat, test_warp_mat, rtol = 1e-4)

class TestFindSatellitesByDist(unittest.TestCase):
	'''
	Tests finding satellites based on distance cutoff
	'''

	@classmethod
	def setUpClass(self):
		self.colony_tracker = \
			track_colonies.ColonyTracker()

	def test_find_sat_by_dist(self):
		parent_candidate_df = pd.DataFrame({
			'cX': [11, 40, 55.4, 80, 101.3],
			'cY': [21.5, 21.5, 30, 100, 20],
			'major_axis_length': [30, 18, 18, 9, 21]},
			index = [3, 2, 1, 15, 16])
		# first colony should match both first and second parent
		# second and fifth colony match no parent colony
		# third colony matches only 5th parent colony
		# fourth and sixth colonies match only 3rd parent colony
		sat_candidate_df = pd.DataFrame({
			'cX': [30, 20, 95.5, 51.5, 85, 59],
			'cY': [21.5, 100, 12, 34, 50, 19],
			'major_axis_length': [2, 4, 3, 2, 5, 4]
			},
			index = [21, 32, 43, 54, 11, 103])
		expected_parent_sat_df = pd.DataFrame({
			'satellite_idx': [43, 54, 103],
			'parent_idx': [16, 1, 1]
			})
		test_parent_sat_df = \
			self.colony_tracker._find_satellites_by_dist(
				parent_candidate_df, sat_candidate_df)
		assert_frame_equal(expected_parent_sat_df, test_parent_sat_df)

	def test_find_sat_by_dist_no_match(self):
		parent_candidate_df = pd.DataFrame({
			'cX': [11, 40, 55.4, 80, 101.3],
			'cY': [21.5, 21.5, 30, 100, 20],
			'major_axis_length': [30, 18, 18, 9, 21]},
			index = [3, 2, 1, 15, 16])
		# first colony should match both first and second parent
		# second and fifth colony match no parent colony
		# third colony matches only 5th parent colony
		# fourth and sixth colonies match only 3rd parent colony
		sat_candidate_df = pd.DataFrame({
			'cX': [20, 85],
			'cY': [100, 50],
			'major_axis_length': [4, 5]
			},
			index = [32, 11])
		expected_parent_sat_df = pd.DataFrame({
			'satellite_idx': [],
			'parent_idx': []
			})
		test_parent_sat_df = \
			self.colony_tracker._find_satellites_by_dist(
				parent_candidate_df, sat_candidate_df)
		assert_frame_equal(expected_parent_sat_df, test_parent_sat_df,
			check_dtype=False)

	def test_find_sat_by_dist_no_sat(self):
		parent_candidate_df = pd.DataFrame({
			'cX': [11, 40],
			'cY': [21.5, 21.5],
			'major_axis_length': [30, 18]},
			index = [3, 2])
		sat_candidate_df = pd.DataFrame({
			'cX': [],
			'cY': [],
			'major_axis_length': []
			},
			index = [])
		expected_parent_sat_df = pd.DataFrame({
			'satellite_idx': [],
			'parent_idx': []
			})
		test_parent_sat_df = \
			self.colony_tracker._find_satellites_by_dist(
				parent_candidate_df, sat_candidate_df)
		assert_frame_equal(expected_parent_sat_df, test_parent_sat_df,
			check_dtype=False)

	def test_find_sat_by_dist_no_parent(self):
		parent_candidate_df = pd.DataFrame({
			'cX': [],
			'cY': [],
			'major_axis_length': []},
			index = [])
		sat_candidate_df = pd.DataFrame({
			'cX': [30, 20],
			'cY': [21.5, 100],
			'major_axis_length': [2, 4]
			},
			index = [21, 32])
		expected_parent_sat_df = pd.DataFrame({
			'satellite_idx': [],
			'parent_idx': []
			})
		test_parent_sat_df = \
			self.colony_tracker._find_satellites_by_dist(
				parent_candidate_df, sat_candidate_df)
		assert_frame_equal(expected_parent_sat_df, test_parent_sat_df,
			check_dtype=False)

	def test_find_sat_by_dist_real_data(self):
		parent_candidate_df = satellite_prop_df.loc[[31,32,35,36,37]]
		sat_candidate_df = satellite_prop_df.loc[[33,34]]
		expected_parent_sat_df = pd.DataFrame({
			'satellite_idx': [34],
			'parent_idx': [35]
			})
		test_parent_sat_df = \
			self.colony_tracker._find_satellites_by_dist(
				parent_candidate_df, sat_candidate_df)
		assert_frame_equal(expected_parent_sat_df, test_parent_sat_df)

class TestAggregateByParent(unittest.TestCase):
	'''
	Tests aggregation of colony_tracker.active_property_df by
	parent_colony
	'''

	def setUp(self):
		self.colony_tracker = \
			track_colonies.ColonyTracker()

	def test_aggregation(self):
		self.colony_tracker.tracking_col_name = 'time_tracking_id'
		self.colony_tracker.active_col_prop_df = pd.DataFrame({
			'phase_num': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2],
			'timepoint': [1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 3, 3, 1, 2, 1, 2, 3],
			'parent_colony': [
				'a', 'b', 'c', 'a', 'a', 'b', 'a', 'b', 'b', 'b', 'b', 'f',
				'x', 'x', 'x', 'x', 'y'
				],
			'xy_pos_idx': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
			'label': np.arange(1, 18).astype(str),
			'area': [
				1, 2.1, 3.21, 5.4321, 4.321, 6.09, 7.1, 8.19, 9.13, 10,
				11.5, 12.43, 13.67, 14.85, 15.69, 16.9, 17
				],
			'perimeter': np.arange(1,18)*3+.4,
			'time_tracking_id': [
				'a', 'b', 'c', 'a', 'd', 'b', 'a', 'b', 'e', 'b', 'b', 'f',
				'x', 'x', 'x', 'x', 'y'
				]
			}, index = range(100,117))
		expected_property_df = pd.DataFrame({
			'phase_num': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2],
			'timepoint': [1, 1, 1, 2, 2, 1, 1, 2, 3, 3, 1, 2, 1, 2, 3],
			'xy_pos_idx': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
			'perimeter':
				np.array(
					[1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]
					)*3+.4,
			'time_tracking_id': [
				'a', 'b', 'c', 'a', 'b', 'a', 'b', 'b', 'b', 'f',
				'x', 'x', 'x', 'x', 'y'
				],
			'label': [
				'1', '2', '3', '4;5', '6', '7', '8', '9;10', '11', '12','13',
				'14', '15', '16', '17'
				],
			'area': [
				1, 2.1, 3.21, 4.321+5.4321, 6.09, 7.1, 8.19, 9.13+10,
				11.5, 12.43, 13.67, 14.85, 15.69, 16.9, 17
					]
			},
			index = [
				100, 101, 102, 103, 105, 106, 107, 109, 110, 111, 112, 113,
				114, 115, 116
				])
		test_property_df = self.colony_tracker._aggregate_by_parent()
		assert_frame_equal(expected_property_df, test_property_df,
			check_index_type = False)

class TestIDSatellites(unittest.TestCase):
	'''
	Tests satellite identification and matching with parents
	'''

	def setUp(self):
		self.colony_tracker = \
			track_colonies.ColonyTracker()

	def test_id_sat_real_data(self):
		self.colony_tracker.active_col_prop_df = satellite_prop_df
		self.colony_tracker.tracking_col_name = 'time_tracking_id'
		match_df_filt = pd.DataFrame({
			'curr_im_colony': [24,25,28,29,30],
			'next_im_colony': [31,32,35,36,37]
			})
		expected_parent_sat_df = pd.DataFrame({
			'satellite_idx': [34],
			'parent_idx': [35]
			})
		test_parent_sat_df = \
			self.colony_tracker._id_satellites(
				satellite_prop_df[satellite_prop_df.timepoint == 11],
				match_df_filt)
		assert_frame_equal(expected_parent_sat_df, test_parent_sat_df)