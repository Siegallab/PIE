#!/usr/bin/python

'''
Tracks colonies through time in a single imaging field
'''

import cv2
import itertools
import numpy as np
import pandas as pd
from PIE import image_properties, analysis_configuration

class SinglePhaseSinglePosCompiler(object):
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
			curr_analysis_config.fluor_channel_df.itertuples():
			# retrieve fluorescent image
			fluor_im, _, _ = \
				curr_analysis_config.get_image(timepoint,
					fluor_row.fluor_channel_label)
			# measure fluorescences
			image_analyzer.measure_fluorescence(fluor_im,
				fluor_row.fluor_channel_column_name,
				fluor_row.fluor_threshold)

	def analyze_timepoint_ims(self):
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
				# check that image size is the same for all images
				if image.shape[0] != self.analysis_config.im_height or \
					image.shape[1] != self.analysis_config.im_width:
					raise ValueError(
						'It looks like not all image sizes are the same!'
						'Compare ' + image_name + ' with ' +
						self.analysis_config.size_ref_im)
				image_analyzer = image_properties.ImageAnalyzer(image,
					image_name, self.analysis_config.phase_output_path, 
					self.analysis_config.main_channel_imagetype,
						self.analysis_config.hole_fill_area,
						self.analysis_config.cleanup,
						self.analysis_config.max_proportion_exposed_edge,
						self.analysis_config.cell_intensity_num,
						self.analysis_config.save_extra_info,
						max_col_num = self.analysis_config.max_colony_num,
						write_col_props_file = False)
				image_analyzer.process_image()
				# if there are fluorescent channels, loop through them and
				# measure fluorescence of each colony
				if not self.analysis_config.fluor_channel_df.empty:
					self._perform_fluor_measurements(self.analysis_config,
						image_analyzer, timepoint)
				# if there's a postphase_analysis_config, perform
				# postphase fluor measurements (at every timepoint)
				if self.postphase_analysis_config is not None:
					self._perform_fluor_measurements(
						self.postphase_analysis_config, image_analyzer,
						None)
				# retrieve colony_property_df
				colony_property_df = \
					image_analyzer.get_colony_property_df_to_save()
				# fill in info for current timepoint
				colony_property_df['main_image_name'] = image_name
				colony_property_df['xy_pos_idx'] = \
					self.analysis_config.xy_position_idx
				colony_property_df['timepoint'] = timepoint
				colony_property_df['time_in_seconds'] = image_time
				colony_property_df['phase_num'] = self.analysis_config.phase_num
				# append current timepoint to combined info
				colony_prop_across_time_list.append(colony_property_df)
		# concatenate all colony property dataframes to single list
		if colony_prop_across_time_list:
			phase_colony_data_untracked = pd.concat(colony_prop_across_time_list)
		else:
			phase_colony_data_untracked = pd.DataFrame()
		return(phase_colony_data_untracked)

class ColonyTracker(object):
	'''
	Generic class for tracking colonies based on dataframe of colony 
	properties in an image and the subsequent image

	The details of the tracking differ from the original matlab code 
	when it comes to colony splintering/merging, as well as the fact 
	that here, image registration occurs between every consecutive 
	timepoint

	In addition, dealing with filters by min_growth_time, appearance in 
	first timepoint, etc is outsourced to growth rate functions

	IMPORTANT:

	tracking IDs created within this class are only unique 
	for the colony property dataframes being passed to it (which may 
	be only for a single xy position, etc), NOT across the whole 
	experiment
	'''
	def __init__(self):
		# initialize dictionary for time-tracked col property dfs
		self.single_phase_col_prop_df_dict = dict()
		# set proportion of major axis length that satellite must be
		# from colony center to be considered a satellite
		self._max_sat_major_axis_prop = .7

	def _find_centroid_transform(self, curr_im_data, next_im_data):
		'''
		Takes in dataframes of data for the current image and the
		following image
		Returns a rigid-body affine transformation matrix M that
		transforms centroid positions in curr_im_data to centroid
		positions in next_im_data
		'''
		# make matrices of centroids
		curr_centroids = np.float32(curr_im_data[['cX', 'cY']].to_numpy())
		next_centroids = np.float32(next_im_data[['cX', 'cY']].to_numpy())
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
		if len(matches)>0 and len(matches[0])>1:
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

	def _register_images(self, curr_im_data, next_im_data):
		'''
		Finds the rigid affine transform between the centroids from
		curr_im_data to next_im_data, applies it to the centroids and
		bounds of curr_im_data, and returns the transformed curr_im_data
		NB: The bound transformation for rotations is not totally
		correct, since the bounding box is rotated and a new bounding
		box encompassing the old, transformed one is created; this is
		larger (or smaller!) than if a new bounding box were calculated 
		on the warped image, but with low levels of rotation expected 
		in most images, this should not be a big problem for tracking
		'''
		### !!! NEEDS UNITTEST
		warp_mat = self._find_centroid_transform(curr_im_data, next_im_data)
		# id bounding box and centroids
		data_to_warp = curr_im_data.copy()
		data_to_warp['bb_x_right'] = \
			data_to_warp['bb_width'] + data_to_warp['bb_x_left']
		data_to_warp['bb_y_bottom'] = \
			data_to_warp['bb_height'] + data_to_warp['bb_y_top']
		curr_im_bb_small = \
			np.float32(data_to_warp[['bb_x_left', 'bb_y_top']].to_numpy())
		curr_im_bb_large = \
			np.float32(data_to_warp[['bb_x_right', 'bb_y_bottom']].to_numpy())
		curr_im_centroids = np.float32(data_to_warp[['cX', 'cY']].to_numpy())
		# warp centroids and bb
		warped_centroids = \
			np.squeeze(np.float32(cv2.transform(curr_im_centroids[np.newaxis],
				warp_mat)))
		warped_bb_small = np.squeeze(
			np.float32(cv2.transform(curr_im_bb_small[np.newaxis], warp_mat)))
		warped_bb_large = np.squeeze(
			np.float32(cv2.transform(curr_im_bb_large[np.newaxis], warp_mat)))
		# calculate and populate warped current image df
		warped_curr_im_data = curr_im_data.copy()
		warped_curr_im_data[['cX', 'cY']] = warped_centroids
		warped_curr_im_data[['bb_x_left', 'bb_y_top']] = \
			np.round(warped_bb_small).astype(int)
		warped_curr_im_data[['bb_width', 'bb_height']] = \
			np.round(warped_bb_large- warped_bb_small).astype(int)
		return(warped_curr_im_data)

	def _get_overlap(self, curr_im_data, next_im_data):
		'''
		Makes overlap df of colonies between two images
		Identifies pairs of colonies between images for which the
		centroid of one falls inside the other's bounding box
		'''
		# perform image registration
		if self.perform_registration:
			curr_im_data_reg = \
				self._register_images(curr_im_data, next_im_data)
		else:
			curr_im_data_reg = curr_im_data.copy()
		# get matrices of absolute differences between x and y
		# centroids of the colonies in the two images
		cX_diff_mat = np.abs(np.subtract.outer(
			curr_im_data_reg['cX'].to_numpy(),
			next_im_data['cX'].to_numpy()))
		cY_diff_mat = np.abs(np.subtract.outer(
			curr_im_data_reg['cY'].to_numpy(),
			next_im_data['cY'].to_numpy()))
		# calculate whether the other image's centroid falls
		# within each respective image's bounding box bounds
		curr_im_cX_bbox_overlap = (cX_diff_mat.T <
			curr_im_data_reg['bb_width'].to_numpy()/2).T
		curr_im_cY_bbox_overlap = (cY_diff_mat.T <
			curr_im_data_reg['bb_height'].to_numpy()/2).T
		next_im_cX_bbox_overlap = \
			cX_diff_mat < next_im_data['bb_width'].to_numpy()/2
		next_im_cY_bbox_overlap = \
			cY_diff_mat < next_im_data['bb_height'].to_numpy()/2
		# for each image, determine whether bounding box
		# encapsulates other image's centroid
		curr_bbox_next_centroid_overlap = \
			np.logical_and(curr_im_cX_bbox_overlap,
				curr_im_cY_bbox_overlap)
		next_bbox_curr_centroid_overlap = \
			np.logical_and(next_im_cX_bbox_overlap,
				next_im_cY_bbox_overlap)
		# create matrix of overlaps between images
		# overlaps occur if a colony's bounding box in either
		# image includes a colony's centroid in the other
		# image
		# this is the incidence matrix in colonies between
		# subsequent images
		overlap_mat = np.logical_or(curr_bbox_next_centroid_overlap,
			next_bbox_curr_centroid_overlap)
		# convert to dataframe with rows (indices) and columns labeled
		# with indices of colonies from original colony property df
		overlap_df = pd.DataFrame(overlap_mat,
			index = curr_im_data_reg.index,
			columns = next_im_data.index)
		return(overlap_df)

	def _id_colony_match(self, curr_im_data, next_im_data):
		'''
		Identifies colonies from next_im_data that match to colonies in
		curr_im_data

		Returns df of matches (by idx in original colony property df)
		'''
		# get df with labeled overlap matrix between current image and
		# the subsequent one
		overlap_df = self._get_overlap(curr_im_data, next_im_data)
		# identify all colony matches
		(curr_im_idx, next_im_idx) = np.where(overlap_df)
		match_df = pd.DataFrame(
			{'curr_im_colony': overlap_df.index[curr_im_idx],
			'next_im_colony': overlap_df.columns[next_im_idx]})
		return(match_df)

	def _get_matching_indices(self, df, col_name, val_to_match):
		'''
		Returns list of indices of df where col_name matches
		val_to_match, even if val_to_match is NaN
		'''
		### !!! NEEDS UNITTEST
		if isinstance(val_to_match, float) and np.isnan(val_to_match):
			match_bool = df[col_name].isnull()
		else:
			match_bool = df[col_name] == val_to_match
		indices = df.index[match_bool].to_list()
		return(indices)

	def _match_by_parent(self, match_df, curr_im_data, next_im_data):
		'''
		Removes any matches in which a satellite matches its parent

		Adds all matches based on shared parentage (i.e. if colony
		whose parent is A matches a colony whose parent is B, all
		colonies whose parent is A match all colonies whose parent is B)
		
		Returns df of matches (by idx in original colony property df)
		'''
		### !!! NEEDS UNITTEST
		match_df_parent = pd.DataFrame(
			{'curr_im_parent':
				curr_im_data.loc[match_df.curr_im_colony]['parent_colony'].values,
			'next_im_parent':
				next_im_data.loc[match_df.next_im_colony]['parent_colony'].values})
		# remove cases of colonies merging with their parents
		match_df_parent = match_df_parent[
			match_df_parent.curr_im_parent != match_df_parent.next_im_parent
			].drop_duplicates()
		# add all combos of parent-parent and parent-satellite matches
		# based on parent-parent matches to match_df
		match_df_list = []
		for _, row in match_df_parent.iterrows():
			curr_colonies = \
				self._get_matching_indices(
					curr_im_data,
					self.tracking_col_name,
					row.curr_im_parent
					)
			next_colonies = \
				self._get_matching_indices(
					next_im_data,
					self.tracking_col_name,
					row.next_im_parent
					)
			curr_match_df = pd.DataFrame(
				list(itertools.product(
					*[curr_colonies, next_colonies])),
				columns = ['curr_im_colony', 'next_im_colony'])
			match_df_list.append(curr_match_df)
		if len(match_df_list) > 0:
			match_df_by_parent = pd.concat(match_df_list, sort = False)
		else:
			match_df_by_parent = \
				pd.DataFrame(columns = ['curr_im_colony', 'next_im_colony'])
		return(match_df_by_parent)

	def _tag_merged_colonies(self, match_df):
		'''
		Identifes merge events between two subsequent images
		(When a single colony in subsequent image matches multiple
		colonies in current image), including both parts of a
		broken-up colony if one of those parts merges with another
		colony
		Set self.tracking_col_name in merged colonies to np.nan
		This means that if a merged colony is tracked in a subsequent
		image, it will inherit np.nan as its unique tracking id
		As a result, merged colonies are no longer tracked, which is the
		desired behavior
		'''
		### !!! NEEDS UNITTEST
		# id colony indices that appear twice in next_im_colony_id
		# column of match_df, i.e. colonies in the next image that
		# match to two different colonies in the current image
		[next_im_col_matching, next_match_count] = \
			np.unique(match_df['next_im_colony'], return_counts=True)
		merged_colonies_direct = next_im_col_matching[next_match_count > 1]
		# if a colony breaks into two, and then one part of it merges
		# with another colony, we want to tag both the merged and
		# non-merged part as merged
		colonies_that_merge = \
			match_df.curr_im_colony[
				match_df['next_im_colony'].isin(merged_colonies_direct)]
		colonies_that_merge_bool = \
			match_df['curr_im_colony'].isin(colonies_that_merge)
		merged_colonies = match_df.next_im_colony[colonies_that_merge_bool]
		# set self.tracking_col_name and parent_colony to NaN for merged
		# colonies in self.active_col_prop_df
		self.active_col_prop_df.loc[
			merged_colonies, self.tracking_col_name] = np.nan
		self.active_col_prop_df.loc[
			merged_colonies, 'parent_colony'] = np.nan
		# return a version of match_df that is missing merged colonies
		match_df_nonmerging = match_df[~colonies_that_merge_bool]
		return(match_df_nonmerging)

	def _resolve_splits(self, match_df_nonmerging):
		'''
		If a single colony in the current image matches with two
		colonies in the subsequent image (e.g. due to breaking up),
		treat the largest of the next image colonies as the matching
		colony
		This step should be done *after* removing merging colonies
		'''
		### !!! NEEDS UNITTEST
		# id colony indices that appear twice in curr_im_colony_id
		# column of match_df_nonmerging, i.e. colonies in the current
		# image that match to two different colonies in the next
		# image
		[curr_im_col_matching, curr_match_count] = \
			np.unique(match_df_nonmerging['curr_im_colony'],
				return_counts=True)
		split_colonies = curr_im_col_matching[curr_match_count > 1]
		# create a dataframe that excludes split colonies
		match_df_no_splits = \
			match_df_nonmerging[
				~match_df_nonmerging.curr_im_colony.isin(split_colonies)]
		# loop through split colonies and add the match that has
		# the largest area in the next image to match_df_filtered
		split_colony_match_list = [match_df_no_splits]
		for curr_colony in split_colonies:
			curr_match_df = \
				match_df_nonmerging[match_df_nonmerging.curr_im_colony ==
					curr_colony].copy()
			current_match_areas = self.active_col_prop_df.loc[
				curr_match_df.next_im_colony.to_list(), 'area'].to_numpy()
			curr_match_df['area'] = current_match_areas
			match_to_keep = \
				curr_match_df[curr_match_df.area == curr_match_df.area.max()]
			# in case two areas of split colonies are identical,
			# keep the first
			match_to_keep = match_to_keep.iloc[[0]]
			split_colony_match_list.append(
				match_to_keep.drop(columns = ['area']))
		# combine selected split colonies with
		# match_df_no_splits into single df
		match_df_filtered = \
			pd.concat(split_colony_match_list, sort = False)
		return(match_df_filtered)

	def _find_satellites_by_dist(self, parent_candidate_df, sat_candidate_df):
		'''
		Returns df with indices of sat_candidate_df rows in which
		satellite colony is within self._max_sat_major_axis_prop of a
		single colony from parent_candidate_df, and indices of
		corresponding parent colonies
		'''
		# create distance matrix with parent colonies along columns
		# and satellite colonies along rows
		if parent_candidate_df.shape[0] == 0 or sat_candidate_df.shape[0] == 0:
			sat_indices = []
			parent_indices = []
		else:
			x_dist = \
				parent_candidate_df.cX.to_numpy() - \
				sat_candidate_df.cX.to_numpy()[np.newaxis].T
			y_dist = \
				parent_candidate_df.cY.to_numpy() - \
				sat_candidate_df.cY.to_numpy()[np.newaxis].T
			dist_mat = np.sqrt(x_dist**2 + y_dist**2)
			# find positions in dist_mat within
			# self._max_sat_major_axis_prop of major_axis_length of parent
			cutoff_dist_array = \
				parent_candidate_df.major_axis_length.to_numpy()*\
				self._max_sat_major_axis_prop
			within_cutoff_mat = dist_mat <= cutoff_dist_array
			# find the number of 'parents' each satellite matches to
			parent_colony_num = np.sum(within_cutoff_mat, axis = 1)
			# only real 'satellites' are colonies that match a single parent
			real_sat_pos_list = parent_colony_num == 1
			# find position corresponding to parent of each real satellite
			parent_pos_list = \
				np.argmax(within_cutoff_mat[real_sat_pos_list,:], axis = 1)
			sat_indices = sat_candidate_df.index[real_sat_pos_list]
			parent_indices = parent_candidate_df.index[parent_pos_list]
		parent_sat_df = pd.DataFrame(
			{'satellite_idx': sat_indices,
			'parent_idx': parent_indices})
		return(parent_sat_df)

	def _id_satellites(self, next_im_data, match_df_filtered):
		'''
		Identifies colonies in next_im_data that are satellites of
		other colonies

		Must be fone after tagging merged colonies and resolving splits

		Satellites are colonies that:
		- 	first appear in next_im_data (i.e. have no match in
			match_df_filtered, and have not been removed due to merging)
		- 	are within self._max_sat_major_axis_prop of only one other
			colony
		'''
		# find colonies in next_im_data with no match (post-removing
		# minor colony split products) in previous timepoint and that
		# have not been filtered out as merge products
		unmatched_colonies = next_im_data.loc[
			~next_im_data.index.isin(match_df_filtered.next_im_colony) & 
			~pd.isnull(self.active_col_prop_df.loc[
				next_im_data.index][self.tracking_col_name])
			]
		# for now, allow all colonies that are not satellite candidates
		# to be parent candidates; afterwards, filter out parents that
		# were not successfully tracked from the previous timepoint
		parent_candidates = next_im_data[
			~next_im_data.index.isin(unmatched_colonies.index)]
		# get dataframe of parent-satellite pair indices from
		# self.active_col_prop_df
		parent_sat_df = \
			self._find_satellites_by_dist(
				parent_candidates,
				unmatched_colonies
				)
		# remove any parent colonies that are satellites to colonies
		# that weren't matched successfully to previous timepoint (e.g.
		# merged colonies)
		parent_sat_df_filt = parent_sat_df[
			parent_sat_df.parent_idx.isin(match_df_filtered.next_im_colony)]
		return(parent_sat_df_filt)

	def _replace_vals_by_keydict(self, replace_keys, replace_vals,
		col_to_replace):
		'''
		Replaces replace_keys in col_to_replace of
		self.active_col_prop_df with replace_vals
		'''
		if len(replace_keys) > 0 and len(replace_vals) > 0:
			replace_dict = pd.Series(
				replace_vals,
				index=replace_keys
				).to_dict()
		else:
			replace_dict = dict()
		self.active_col_prop_df.replace(
			{col_to_replace: replace_dict},
			inplace = True)

	def _replace_tracking_col_by_match(self, match_df, curr_im_data,
		next_im_data):
		'''
		Uses match_df to replace tracking_col_name and parent_coloy in
		self.active_col_prop_df from every timepoint that have the
		values in match_df.next_im_colony with values from
		match_df.curr_im_colony
		'''
		### !!! NEEDS UNITTEST
		# replace tracking id values based on match_df
		replace_vals_track = \
			curr_im_data.loc[match_df.curr_im_colony.values]\
				[self.tracking_col_name].to_list()
		replace_keys_track = \
			next_im_data.loc[match_df.next_im_colony.values]\
				[self.tracking_col_name].to_list()
		self._replace_vals_by_keydict(
			replace_keys_track, replace_vals_track, self.tracking_col_name)
		# replace parent_colony values based on match_df
		replace_vals_parent = \
			curr_im_data.loc[match_df.curr_im_colony.values]\
				['parent_colony'].to_list()
		replace_keys_parent = \
			next_im_data.loc[match_df.next_im_colony.values]\
				['parent_colony'].to_list()
		self._replace_vals_by_keydict(
			replace_keys_parent, replace_vals_parent, 'parent_colony')

	def _replace_parent_for_sat(self, parent_sat_df):
		'''
		Uses parent_sat_df to replace parent_colony in
		self.active_col_prop_df from every timepoint that have the
		values in parent_sat_df.satellite_idx with values from
		parent_sat_df.parent_idx
		'''
		### !!! NEEDS UNITTEST
		replace_vals = \
			self.active_col_prop_df.loc[
				parent_sat_df.parent_idx.values
				]['parent_colony'].to_list()
		replace_keys = \
			self.active_col_prop_df.loc[
				parent_sat_df.satellite_idx.values
				]['parent_colony'].to_list()
		self._replace_vals_by_keydict(
			replace_keys, replace_vals, 'parent_colony')

	def _track_single_im_pair(self, curr_im_data, next_im_data):
		'''
		Performs full tracking between curr_im_data and next_im_data,
		and records results in self.active_col_prop_df
		'''
		# identify all matches between colonies in the two images
		match_df = self._id_colony_match(curr_im_data, next_im_data)
		# modify matches so that all colonies with shared parent share
		# matches, and matches between colonies with same parent are
		# removed
		match_df_by_parent = \
			self._match_by_parent(match_df, curr_im_data, next_im_data)
		# set self.tracking_col_name in merged colonies to NaN, and get
		# back match_df without colonies that merge into others
		match_df_nonmerging = self._tag_merged_colonies(match_df_by_parent)
		# for colonies that break into multiple pieces, track the
		# largest piece as the main colony
		match_df_filtered = self._resolve_splits(match_df_nonmerging)
		# identify satellite colonies and their parents in next_im_data
		parent_sat_df_filt = \
			self._id_satellites(next_im_data, match_df_filtered)
		# for colonies with a direct match between subsequent
		# images, set self.tracking_col_name in all images to the
		# self.tracking_col_name from current_im_data
		self._replace_tracking_col_by_match(match_df_filtered, curr_im_data,
			next_im_data)
		# for colonies in next_im that are a satellite, set the
		# parent_colony of every colony that shares their parent_colony
		# to the matching parent value from parent_sat_df_filt
		# This is important in cross-phase tracking, when a colony that
		# has already been tracked independently across multiple
		# timepoints turns out to be a satellite based on tracking from
		# a previous phase (in this case, these satellites will not
		# receive independent growth rates in the corresponding
		# phase's growth rate analysis)
		self._replace_parent_for_sat(parent_sat_df_filt)

	def _set_up_satellite_track(self):
		'''
		Sets up columns necessary for tracking satellites
		'''
		# initialize parent_colony column
		self.active_col_prop_df['parent_colony'] = \
			self.active_col_prop_df[self.tracking_col_name]

	def _aggregate_by_parent(self):
		'''
		Aggregates data in self.active_col_prop_df by parent in the
		following way:
		-	label across all colonies sharing same parent gets joined
			with semicolon
		-	area is summed across all colonies sharing same parent
		-	all other colony properties are taken from the parent
		-	parent_colony column is dropped
		'''
		# set aside parent colonies only
		parent_property_df = \
			self.active_col_prop_df.loc[
				self.active_col_prop_df['parent_colony'] == 
				self.active_col_prop_df[self.tracking_col_name]
				].copy()
		# remove properties to be aggregated
		parent_property_df.drop(
				columns=['label', 'area'],
				inplace = True)
		# set up df for aggregated properties
		agg_property_df_group = \
			self.active_col_prop_df[[
					'phase_num', 'timepoint', 'parent_colony', 'xy_pos_idx',
					'label', 'area']
				].groupby(
					['phase_num', 'timepoint', 'parent_colony', 'xy_pos_idx']
					)
		agg_property_df = agg_property_df_group.agg({
			'label': ';'.join,
			'area': np.sum
			})
		property_df = parent_property_df.join(agg_property_df,
			on = ['phase_num', 'timepoint', 'parent_colony', 'xy_pos_idx'])
		property_df.drop(columns = ['parent_colony'], inplace = True)
		return(property_df)

	def match_and_track_across_time(self, phase_num, colony_prop_df,
		perform_registration = True):
		'''
		Identifies matching colonies between subsequent timepoints
		Performs equivalent role to ColonyAreas_mod in original matlab
		code, but with image registration; see comments on class for
		differences
		'''
		self.perform_registration = perform_registration
		### !!! NEEDS UNITTEST
		# set the tracking column we'll be using to track across time
		self.tracking_col_name = 'time_tracking_id'
		# set active_col_prop_df to colony_prop_df
		self.active_col_prop_df = colony_prop_df.copy()
		# reset indices in self.active_col_prop_df to be consecutive
		# integers, to make sure each row has a unique index that can be
		# used as an identifier
		self.active_col_prop_df.reset_index(inplace = True, drop=True)
		# identify xy positions
		_xy_positions = \
			np.sort(np.unique(self.active_col_prop_df.xy_pos_idx))
		# identify timepoints
		_curr_phase_timepts = \
			np.sort(np.unique(self.active_col_prop_df.timepoint))
		# populate tracking column with unique IDs
		self.active_col_prop_df[self.tracking_col_name] = \
			["phase_{}_xy{}_col{}".format(phase_num, xy_pos, col_id) for \
				phase_num, xy_pos, col_id in \
				zip(self.active_col_prop_df.phase_num,
					self.active_col_prop_df.xy_pos_idx,
					self.active_col_prop_df.index)]
		# set up columns for tracking satellites
		self._set_up_satellite_track()
		if len(_curr_phase_timepts) > 1:
			# loop through xy positions
			for curr_xy_pos in _xy_positions:
				# loop through timepoints
				for curr_timepoint, next_timepoint in \
					zip(_curr_phase_timepts[:-1], _curr_phase_timepts[1:]):
					# get df of which colony in curr_timepoint matches
					# which colony in next_timepoint
					###
					# get colony properties at current timepoint
					curr_im_data = \
						self.active_col_prop_df[
							(self.active_col_prop_df.timepoint ==
								curr_timepoint) &
							(self.active_col_prop_df.xy_pos_idx ==
								curr_xy_pos)]
					# get colony properties at next timepoint
					next_im_data = \
						self.active_col_prop_df[
							(self.active_col_prop_df.timepoint ==
								next_timepoint) &
							(self.active_col_prop_df.xy_pos_idx ==
								curr_xy_pos)]
					self._track_single_im_pair(curr_im_data, next_im_data)
		# convert time tracking id column to categorical
		self.active_col_prop_df[self.tracking_col_name] = \
			self.active_col_prop_df[self.tracking_col_name].astype(
				'category')
		# aggregate data by parent_colony
		parent_agg_prop_df = self._aggregate_by_parent()
		# add data for this phase to dict
		self.single_phase_col_prop_df_dict[phase_num] = \
			parent_agg_prop_df
		return(parent_agg_prop_df)

	def match_and_track_across_phases(self, perform_registration = True):
		'''
		Identifies matching colonies between last timepoint of each
		phase and the first timepoint of the subsequent phase
		Performs equivalent role to ColonyAreas_mod in original matlab
		code, but see comments on class for differences
		'''
		### !!! NEEDS UNITTEST
		self.perform_registration = perform_registration
		# concatenate time-tracked colony property dfs to create df for
		# tracking colonies across phases
		indiv_phase_dfs = list(self.single_phase_col_prop_df_dict.values())
		if indiv_phase_dfs:
			self.active_col_prop_df = pd.concat(indiv_phase_dfs, sort = False)
			# set the tracking column we'll be using to track across phases
			self.tracking_col_name = 'cross_phase_tracking_id'
			# reset indices in self.active_col_prop_df to be consecutive
			# integers, to make sure each row has a unique index that can be
			# used as an identifier
			self.active_col_prop_df.reset_index(inplace = True, drop=True)
			# identify xy positions
			_xy_positions = \
				np.sort(np.unique(self.active_col_prop_df.xy_pos_idx))
			# identify phases
			phase_list = \
				np.sort(np.array(list(
					self.single_phase_col_prop_df_dict.keys())))
			# check that phases are unique
			if np.unique(phase_list).size < phase_list.size:
				raise IndexError('phase list contains non-unique items: ' +
					np.array_str(phase_list))
			# populate tracking column with unique IDs
			# it's important to use time_tracking_id within each phase,
			# so that cross-phase tracking remains consistent across all
			# timepoints within a phase
			# prevents any colonies that merged in a previous phase from
			# being tracked in a later phase
			self.active_col_prop_df['cross_phase_tracking_id'] = \
				self.active_col_prop_df['time_tracking_id']
			# set up columns for tracking satellites
			self._set_up_satellite_track()
			if len(phase_list) > 1:
				# loop through timepoints
				for curr_phase, next_phase in \
					zip(phase_list[:-1], phase_list[1:]):
					# get df of which colony in the last tracked
					# timepoint of curr_phase matches which colony in
					# the last tracked timepoint of next_phase
					###
					# get phase data
					curr_phase_data = self.active_col_prop_df[
						self.active_col_prop_df.phase_num == curr_phase]
					next_phase_data = self.active_col_prop_df[
						self.active_col_prop_df.phase_num == next_phase]
					# find timepoints
					curr_phase_last_tp = np.max(curr_phase_data.timepoint)
					next_phase_first_tp = np.min(next_phase_data.timepoint)
					# loop through xy positions
					for curr_xy_pos in _xy_positions:
						# get colony properties at current timepoint
						curr_im_data = curr_phase_data[
							(curr_phase_data.timepoint == curr_phase_last_tp) &
							(curr_phase_data.xy_pos_idx == curr_xy_pos)]
						# get colony properties at next timepoint
						next_im_data = next_phase_data[
							(next_phase_data.timepoint == next_phase_first_tp) &
							(next_phase_data.xy_pos_idx == curr_xy_pos)]
						self._track_single_im_pair(curr_im_data, next_im_data)
			# convert cross-phase tracking id column to categorical
			self.active_col_prop_df[self.tracking_col_name] = \
				self.active_col_prop_df[self.tracking_col_name].astype(
					'category')
			# aggregate data by parent_colony
			parent_agg_prop_df = self._aggregate_by_parent()
			self.cross_phase_tracked_col_prop_df = \
				parent_agg_prop_df
		else:
			# columns need to be included type to save to parquet later
			self.cross_phase_tracked_col_prop_df = \
				pd.DataFrame(columns = 
					['time_tracking_id', 'phase_num', 'xy_pos_idx',
						'cross_phase_tracking_id','timepoint'])
		return(self.cross_phase_tracked_col_prop_df)

def generate_match_key_df(cross_phase_tracked_col_prop_df):
	'''
	Generates a dataframe that can be used to identify cross phase
	tracking IDs for any other dataframe with xy_pos_idx,
	phase_num, and time_tracking_id info
	'''
	if cross_phase_tracked_col_prop_df.empty:
		match_key_df = pd.DataFrame()
	else:
		match_key_df = \
			cross_phase_tracked_col_prop_df[
				['time_tracking_id', 'xy_pos_idx', 'phase_num',
				'cross_phase_tracking_id']
				].drop_duplicates().reset_index(drop = True)
		match_key_df.dropna(subset=['cross_phase_tracking_id'], inplace = True)
	return(match_key_df)

def track_colonies_single_pos(xy_pos_idx,
	analysis_config_file = None, analysis_config_obj_df = None):
	'''
	Runs image analysis and colony tracking for xy_pos_idx
	Takes either analysis_config_obj_df or analysis_config_file as arg
	'''
	# check that only analysis_config_obj_df or
	# analysis_config_file is passed, and get analysis_config_obj_df
	analysis_config_obj_df = analysis_configuration.check_passed_config(
		analysis_config_obj_df, analysis_config_file)
	# set up colony tracker
	colony_tracker = ColonyTracker()
	# initialize list to check whether to perform image 
	# registration across phases
	cross_phase_registration_bool_list = []
	# track colonies within each phase
	for phase_num in analysis_config_obj_df.index:
		analysis_config = \
			analysis_config_obj_df.at[phase_num, 'analysis_config']
		postphase_analysis_config = \
			analysis_config_obj_df.at[phase_num, 'postphase_analysis_config']
		# perform image analysis on all timepoints for current phase
		# and position
		timelapse_data_compiler = SinglePhaseSinglePosCompiler(
			xy_pos_idx, analysis_config, postphase_analysis_config)
		untracked_phase_pos_data = \
			timelapse_data_compiler.analyze_timepoint_ims()
		if not untracked_phase_pos_data.empty:
			# track colonies across time
			colony_tracker.match_and_track_across_time(
				phase_num,
				untracked_phase_pos_data,
				analysis_config.perform_registration
				)
		cross_phase_registration_bool_list.append(
			analysis_config.perform_registration
			)
	# track colonies across phases
	# only perform registration if perform_registration was true in 
	# all phases
	perform_cross_phase_registration = all(cross_phase_registration_bool_list)
	time_and_phase_tracked_pos_data = \
		colony_tracker.match_and_track_across_phases(
			perform_cross_phase_registration
			)
	# write phase-tracked file to parquet format
	time_and_phase_tracked_pos_data.to_parquet(
		analysis_config.tracked_properties_write_path)
	return(time_and_phase_tracked_pos_data)



