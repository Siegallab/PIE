#!/usr/bin/python

'''
Tracks colonies through time in a single imaging field
'''

import cv2
import numpy as np
import pandas as pd
from PIE import image_properties

class _SinglePhaseSinglePosCompiler(object):
	'''
	Compiles and analyzes results for a single experimental phase at a
	single position
	'''
	def __init__(self, xy_position, analysis_config, postphase_analysis_config = None):
		self.analysis_config = analysis_config
		self.analysis_config.set_xy_position(xy_position)
		self.postphase_analysis_config = postphase_analysis_config
		if self.postphase_analysis_config is not None:
			self.postphase_analysis_config.set_xy_position(xy_position)

	def _perform_fluor_measurements(self, curr_analysis_config, image_analyzer,
		timepoint):
		'''
		Performs fluorescence measurements using image_analyzer based on
		data in analysis_config
		'''
		# set up background and colony masks
		image_analyzer.set_up_fluor_measurements()
		# loop through fluor channels and analyze each one
		for fluor_row in \
			curr_analysis_config.fluor_channel_df.iterrows():
			# retrieve fluorescent image
			fluor_im, _, _ = \
				curr_analysis_config.get_image(timepoint,
					fluor_row.fluor_channel_label)
			# measure fluorescences
			image_analyzer.measure_fluorescence(fluor_im,
				fluor_row.fluor_channel_column_name,
				fluor_row.fluor_threshold)

	def _analyze_timepoints(self):
		'''
		Runs colony edge detection on main (brightfield/phase contrast)
		images from each timepoint, measures any fluorescent data
		concurrent with those timepoints, populates dataframe of colony
		properties
		'''
		### NEEDS UNITTEST
		colony_prop_across_time_list = []
		for timepoint in self.analysis_config.timepoint_list:
			image, image_name, image_time = \
				self.analysis_config.get_image(timepoint,
					self.analysis_config.main_channel_label)
			if image is not None:
				image_analyzer = image_properties._ImageAnalyzer(image,
					image_name, self.analysis_config.phase_output_path, 
					self.analysis_config.main_channel_imagetype,
						self.analysis_config.hole_fill_area,
						self.analysis_config.cleanup,
						self.analysis_config.max_proportion_exposed_edge,
						self.analysis_config.save_extra_info,
						write_col_props_file = False)
				image_analyzer.process_image()
				# if there are fluorescent channels, loop through them and
				# measure fluorescence of each colony
				if not self.analysis_config.fluor_channel_df.empty:
					self._perform_fluor_measurements(self.analysis_config,
						image_analyzer, timepoint)
				# if you're at the last timepoint, and there's a
				# postphase_analysis_config, perform postphase fluor
				# measurements
				if self.postphase_analysis_config != None and \
					(timepoint == np.max(self.analysis_config.timepoint_list)):
					self._perform_fluor_measurements(
						self.postphase_analysis_config, image_analyzer,
						timepoint)
				# retrieve colony_property_df
				colony_property_df = \
					image_analyzer.get_colony_property_df_to_save()
				# fill in info for current timepoint
				colony_property_df['main_image_name'] = image_name
				colony_property_df['xy_pos_idx'] = \
					self.analysis_config.xy_position_idx
				colony_property_df['timepoint'] = timepoint
				colony_property_df['time_in_seconds'] = image_time
				colony_property_df['phase'] = self.analysis_config.phase
				# append current timepoint to combined info
				colony_prop_across_time_list.append(colony_property_df)
		# concatenate all colony property dataframes to single list
		if colony_prop_across_time_list:
			phase_colony_data_untracked = pd.concat(colony_prop_across_time_list)
		else:
			phase_colony_data_untracked = None
		return(phase_colony_data_untracked)

	def analyze_phase_data(self):
		'''
		Performs analysis on current phase, and returns phase colony
		data with tracking info
		'''
		# process all images and collect colony properties, including
		# fluorescence if applicable
		phase_colony_data_untracked = self._analyze_timepoints()
		if phase_colony_data_untracked is None:
			phase_colony_data_tracked = None
		else:
			# identify colony matches through time
			time_tracker = _TimeTracker(phase_colony_data_untracked)
			phase_colony_data_tracked = time_tracker.match_and_track()
		return(phase_colony_data_tracked)

class _TimeTracker(object):
	'''
	Tracks colonies through time based on dataframe of colony properties
	at each timepoint
	The details of the tracking differ from the original matlab code
	when it comes to colony splintering/merging, as well as the fact
	that here, image registration occurs between every consecutive
	timepoint
	In addition, dealing with filters by min_growth_time, settle_frames,
	etc is outsourced to growth rate functions
	'''
	# TODO: CURRENTLY DOESN'T HANDLE SATELLITES!
	#	need to check how those are *really* dealt with in original code
	#	it kind of seems like if they merge with main colony, it counts
	#	as a merge?
	# TODO: As in original matlab code, 'satellite' colonies (broken-off
	# pieces) are only included in the main colony area at the single
	# timepoint after they break off, and then not tracked after
	# May want to reconsider this in future versions

	def __init__(self, timecourse_colony_prop_df):
		self.timecourse_colony_prop_df = timecourse_colony_prop_df
		self._timepoints = np.sort(np.unique(timecourse_colony_prop_df.timepoint))

	def _find_centroid_transform(self, curr_tp_data, next_tp_data):
		'''
		Takes in dataframes of data for the current timepoint and the
		following timepoint
		Returns a rigid-body affine transformation matrix M that
		transforms centroid positions in curr_tp_data to centroid
		positions in next_tp_data
		'''
		# make matrices of centroids
		curr_centroids = np.float32(curr_tp_data[['cX', 'cY']].to_numpy())
		next_centroids = np.float32(next_tp_data[['cX', 'cY']].to_numpy())
		# initialize brute force matcher
		bf = cv2.BFMatcher()
		# return top two matches for each point
		matches = bf.knnMatch(curr_centroids, next_centroids, k=2)
		# Apply ratio test from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
		# This ensures that only those matches are used for finding the
		# transformation matrix which match their #1 match much closer
		# than they match their #2 match; stringency of this requirement
		# can be increased by decreasing match_ratio_cutoff, at the risk
		# of ending up with too few match points
		match_ratio_cutoff = 0.75
		good = []
		for m,n in matches:
		    if m.distance < match_ratio_cutoff * n.distance:
		        good.append(m)
		if len(good) >= 3:
			curr_centroids_matching = curr_centroids[[a.queryIdx for a in good]]
			next_centroids_matching = next_centroids[[a.trainIdx for a in good]]
			# estimateAffinePartial2D removes outliers in its default state
			# and returns a mask showing which points it didn't remove, but
			# we ignore this mask here
			M, _ = cv2.estimateAffinePartial2D(curr_centroids_matching,
				next_centroids_matching)
		else:
			# can't estimate affine matrix with this few points, assume
			# no transformation
			M = np.float64([[1, 0, 0], [0, 1, 0]])
		return(M)

	def _register_timepoints(self, curr_tp_data, next_tp_data):
		'''
		Finds the rigid affine transform between the centroids from
		curr_tp_data to next_to_data, applies it to the centroids and
		bounds of curr_tp_data, and returns the transformed curr_tp_data
		NB: The bound transformation for rotations is not totally
		correct, since the bounding box is rotated and a new bounding
		box encompassing the old, transformed one is created; this is
		larger than if a new bounding box were calculated on the
		warped image, but with low levels of rotation expected in most
		images, this should not be a big problem for tracking
		'''
		### !!! NEEDS UNITTEST
		warp_mat = self._find_centroid_transform(curr_tp_data, next_tp_data)
		# id bounding box and centroids
		data_to_warp = curr_tp_data.copy()
		data_to_warp['bb_x_right'] = \
			data_to_warp['bb_width'] + data_to_warp['bb_x_left']
		data_to_warp['bb_y_bottom'] = \
			data_to_warp['bb_height'] + data_to_warp['bb_y_top']
		curr_tp_bb_small = \
			np.float32(data_to_warp[['bb_x_left', 'bb_y_top']].to_numpy())
		curr_tp_bb_large = \
			np.float32(data_to_warp[['bb_x_right', 'bb_y_bottom']].to_numpy())
		curr_tp_centroids = np.float32(data_to_warp[['cX', 'cY']].to_numpy())
		# warp centroids and bb
		warped_centroids = \
			np.squeeze(np.float32(cv2.transform(curr_tp_centroids[np.newaxis],
				warp_mat)))
		warped_bb_small = np.squeeze(
			np.float32(cv2.transform(curr_tp_bb_small[np.newaxis], warp_mat)))
		warped_bb_large = np.squeeze(
			np.float32(cv2.transform(curr_tp_bb_large[np.newaxis], warp_mat)))
		# calculate and populate warped current timepoint df
		warped_curr_tp_data = curr_tp_data.copy()
		warped_curr_tp_data[['cX', 'cY']] = warped_centroids
		warped_curr_tp_data[['bb_x_left', 'bb_y_top']] = \
			np.round(warped_bb_small).astype(int)
		warped_curr_tp_data[['bb_width', 'bb_height']] = \
			np.round(warped_bb_large- warped_bb_small).astype(int)
	#	print(warp_mat)
	#	print(curr_tp_data[['cX', 'cY', 'bb_x_left', 'bb_y_top', 'bb_width', 'bb_height']]-warped_curr_tp_data[['cX', 'cY', 'bb_x_left', 'bb_y_top', 'bb_width', 'bb_height']])
		return(warped_curr_tp_data)

	def _get_overlap(self, curr_timepoint, next_timepoint):
		'''
		Makes overlap df of colonies between two timepoints
		Identifies pairs of colonies between timepoints for which the
		centroid of one falls inside the other's bounding box
		'''
		# get colony properties at current timepoint
		curr_timepoint_data = \
			self.timecourse_colony_prop_df[
				self.timecourse_colony_prop_df.timepoint == curr_timepoint]
		# get colony properties at next timepoint
		next_timepoint_data = \
			self.timecourse_colony_prop_df[
				self.timecourse_colony_prop_df.timepoint == next_timepoint]
		# perform image registration
		curr_timepoint_data_reg = \
			self._register_timepoints(curr_timepoint_data, next_timepoint_data)
		# get matrices of absolute differences between x and y
		# centroids of the colonies in the two timepoints
		cX_diff_mat = np.abs(np.subtract.outer(
			curr_timepoint_data_reg['cX'].to_numpy(),
			next_timepoint_data['cX'].to_numpy()))
		cY_diff_mat = np.abs(np.subtract.outer(
			curr_timepoint_data_reg['cY'].to_numpy(),
			next_timepoint_data['cY'].to_numpy()))
		# calculate whether the other timepoint's centroid falls
		# within each respective timepoint's bounding box bounds
		curr_time_cX_bbox_overlap = (cX_diff_mat.T <
			curr_timepoint_data_reg['bb_width'].to_numpy()/2).T
		curr_time_cY_bbox_overlap = (cY_diff_mat.T <
			curr_timepoint_data_reg['bb_height'].to_numpy()/2).T
		next_time_cX_bbox_overlap = \
			cX_diff_mat < next_timepoint_data['bb_width'].to_numpy()/2
		next_time_cY_bbox_overlap = \
			cY_diff_mat < next_timepoint_data['bb_height'].to_numpy()/2
#		print(curr_time_cX_bbox_overlap.shape)
#		print(curr_time_cX_bbox_overlap)
#		print(next_time_cX_bbox_overlap.shape)
#		print(next_time_cX_bbox_overlap)
		# for each timepoint, determine whether bounding box
		# encapsulates other timepoint's centroid
		curr_bbox_next_centroid_overlap = \
			np.logical_and(curr_time_cX_bbox_overlap,
				curr_time_cY_bbox_overlap)
		next_bbox_curr_centroid_overlap = \
			np.logical_and(next_time_cX_bbox_overlap,
				next_time_cY_bbox_overlap)
#		print(curr_bbox_next_centroid_overlap.shape)
#		print(next_bbox_curr_centroid_overlap.shape)
		# create matrix of overlaps between timepoints
		# overlaps occur if a colony's bounding box at either
		# timepoint includes a colony's centroid of the other
		# timepoint
		# this is the incidence matrix in colonies between
		# subsequent timepoints
		overlap_mat = np.logical_or(curr_bbox_next_centroid_overlap,
			next_bbox_curr_centroid_overlap)
		# convert to dataframe with rows (indices) and columns
		# labeled with indices of colony in
		# self.timecourse_colony_prop_df
		overlap_df = pd.DataFrame(overlap_mat,
			index = curr_timepoint_data_reg.index,
			columns = next_timepoint_data.index)
		return(overlap_df)

	def _id_colony_match(self, curr_timepoint, next_timepoint):
		'''
		Identifies colonies from next_timepoint that match to
		current_timepoint
		Returns df of matches (by idx in timecourse_colony_prop_df)
		'''
		# get df with labeled overlap matrix between current
		# timepoint and the subsequent one in
		# self.timecourse_colony_prop_df
		overlap_df = self._get_overlap(curr_timepoint, next_timepoint)
		# identify all colony matches
		(curr_time_idx, next_time_idx) = np.where(overlap_df)
		match_df = pd.DataFrame(
			{'curr_time_colony': overlap_df.index[curr_time_idx],
			'next_time_colony': overlap_df.columns[next_time_idx]})
		return(match_df)

	def _tag_merged_colonies(self, match_df):
		'''
		Identifes merge events between two subsequent timepoints
		(When a single colony in subsequent timepoint matches multiple
		colonies in current timepoint), including both parts of a
		broken-up colony if one of those parts merges with another
		colony
		Set unique_tracking_id in merged colonies to np.nan
		This means that if a merged colony is tracked in a subsequent
		timepoint, it will inherit np.nan as its unique tracking id
		As a result, merged colonies are no longer tracked, which is the
		desired behavior
		'''
		### !!! NEEDS UNITTEST
		# id colony indices that appear twice in next_time_colony_id
		# column of match_df, i.e. colonies in the next timepoint that
		# match to two different colonies in the current timepoint
		[next_time_col_matching, next_match_count] = \
			np.unique(match_df['next_time_colony'], return_counts=True)
		merged_colonies_direct = next_time_col_matching[next_match_count > 1]
		# if a colony breaks into two, and then one part of it merges
		# with another colony, we want to tag both the merged and
		# non-merged part as merged
		colonies_that_merge = \
			match_df.curr_time_colony[
				match_df['next_time_colony'].isin(merged_colonies_direct)]
		colonies_that_merge_bool = \
			match_df['curr_time_colony'].isin(colonies_that_merge)
		merged_colonies = match_df.next_time_colony[colonies_that_merge_bool]
		# set unique_tracking_id to NaN for merged colonies in
		# self.timecourse_colony_prop_df
		self.timecourse_colony_prop_df.loc[
			merged_colonies, 'unique_tracking_id'] = np.nan
		# return a version of match_df that is missing merged colonies
		match_df_nonmerging = match_df[~colonies_that_merge_bool]
		return(match_df_nonmerging)

	def _resolve_breakups(self, match_df_nonmerging):
		'''
		If a single colony in the current timepoint matches with two
		colonies in the subsequent timepoint (e.g. due to breaking up),
		treat the largest of the next timepoint colonies as the matching
		colony
		This step should be done *after* removing merging colonies
		'''
		### !!! NEEDS UNITTEST
		# id colony indices that appear twice in curr_time_colony_id
		# column of match_df_nonmerging, i.e. colonies in the current
		# timepoint that match to two different colonies in the next
		# timepoint
		[curr_time_col_matching, curr_match_count] = \
			np.unique(match_df_nonmerging['curr_time_colony'],
				return_counts=True)
		splintered_colonies = curr_time_col_matching[curr_match_count > 1]
		# create a dataframe that excludes splintered colonies
		match_df_no_splinters = \
			match_df_nonmerging[
				~match_df_nonmerging.curr_time_colony.isin(splintered_colonies)]
		# loop through splintered colonies and add the match that has
		# the largest area in the next timepoint to match_df_filtered
		splintered_colony_match_list = [match_df_no_splinters]
		for curr_colony in splintered_colonies:
			curr_match_df = \
				match_df_nonmerging[match_df_nonmerging.curr_time_colony ==
					curr_colony].copy()
			current_match_areas = self.timecourse_colony_prop_df.loc[
				curr_match_df.next_time_colony.to_list(), 'area'].to_numpy()
			curr_match_df['area'] = current_match_areas
			match_to_keep = \
				curr_match_df[curr_match_df.area == curr_match_df.area.max()]
			# in case two areas of splintered colonies are identical,
			# keep the first
			match_to_keep = match_to_keep.iloc[[0]]
			splintered_colony_match_list.append(match_to_keep)
		# combine selected splintered colonies with
		# match_df_no_splinters into single df
		match_df_filtered = \
			pd.concat(splintered_colony_match_list, sort = False)
		return(match_df_filtered)

	def match_and_track(self):
		'''
		Identifies matching colonies between subsequent timepoints
		Performs equivalent role to ColonyAreas_mod in original matlab
		code, but with image registration
		'''
		### !!! NEEDS UNITTEST
		# reset indices in self.timecourse_colony_prop_df to be
		# consecutive integers, to make sure each row has a unique
		# index that can be used as an identifier
		self.timecourse_colony_prop_df.reset_index(inplace = True, drop=True)
		# initialize a column to record match of every colony in
		# subsequent timepoint
		self.timecourse_colony_prop_df['unique_tracking_id'] = \
			["{}_{}_col{}".format(phase, xy_pos, col_id) for \
				phase, xy_pos, col_id in \
				zip(self.timecourse_colony_prop_df.phase,
					self.timecourse_colony_prop_df.xy_pos_idx,
					self.timecourse_colony_prop_df.index)]
#		# initialize a column that will keep track of any 'satellites' a
#		# colony has in each timepoint
#		self.timecourse_colony_prop_df['satellite_to'] = \
#			np.nan
		# loop through timepoints
		for curr_timepoint, next_timepoint in \
			zip(self._timepoints[:-1], self._timepoints[1:]):
			# get df of which colony in curr_timepoint matches which
			# colony in next_timepoint
			match_df = self._id_colony_match(curr_timepoint, next_timepoint)
			# set unique_tracking_id in merged colonies to NaN, and get
			# back match_df without colonies that merge into others
			match_df_nonmerging = self._tag_merged_colonies(match_df)
			# for colonies that break into multiple pieces, track the
			# largest piece as the main colony
			match_df_filtered = self._resolve_breakups(match_df_nonmerging)
			# for colonies with a direct match between subsequent
			# timepoints, set unique_tracking_id in next_timepoint to the
			# unique_tracking_id from current_timepoint
			self.timecourse_colony_prop_df.loc[
				match_df_filtered.next_time_colony, 'unique_tracking_id'] = \
				self.timecourse_colony_prop_df.loc[
					match_df_filtered.curr_time_colony,
						'unique_tracking_id'].values
		return(self.timecourse_colony_prop_df)

def track_single_phase_single_position(xy_pos, analysis_config,
	postphase_analysis_config):
	'''
	Runs colony property measurements and tracking on a single position
	for a single phase
	'''
	# set xy position in analysis config and, if it exists,
	# postphase_analysis_config
	analysis_config.set_xy_position(xy_pos)
	if postphase_analysis_config != None:
		postphase_analysis_config.set_xy_position(xy_pos)
	# run image analysis on each timepoint at xy_pos for the phases
	# represented in analysis_config and postphase_analysis_config, and
	# track the identified colonies through time
	single_phase_single_pos_compiler = \
		_SinglePhaseSinglePosCompiler(xy_pos, analysis_config,
			postphase_analysis_config)
	phase_colony_data_tracked = \
		single_phase_single_pos_compiler.analyze_phase_data()
	return(phase_colony_data_tracked)

def track_single_phase_all_positions(analysis_config, postphase_analysis_config):
	phase_data_tracked_list = []
	for xy_pos in analysis_config.xy_position_vector:
		# track colonies in current phase and at current position
		# don't keep output dataframe since it will be written to a
		# file anyways
		curr_pos_phase_colony_data_tracked = \
			track_single_phase_single_position(xy_pos, analysis_config,
				postphase_analysis_config)
		phase_data_tracked_list.append(curr_pos_phase_colony_data_tracked)
	phase_data_tracked_col_prop_df = \
		pd.concat(phase_data_tracked_list, sort = False)
	phase_data_tracked_col_prop_df.to_csv(
		analysis_config.phase_tracked_properties_write_path)
