#!/usr/bin/python

'''
Tracks colonies through time in a single imaging field
'''

import numpy as np
import pandas as pd
from PIE import image_properties

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
			curr_analysis_config.fluor_channel_df.iterrows():
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
	when it comes to colony splitting/merging
	In addition, dealing with filters by min_growth_time, settle_frames,
	etc is outsourced to growth rate functions
	IMPORTANT:
	tracking IDs created within this class are only unique
	for the colony property dataframes being passed to it (which may
	be only for a single xy position, etc), NOT across the whole
	experiment
	'''
	# TODO: CURRENTLY DOESN'T HANDLE SATELLITES!
	#	need to check how those are *really* dealt with in original code
	#	it kind of seems like if they merge with main colony, it counts
	#	as a merge?
	def __init__(self):
		# initialize dictionary for time-tracked col property dfs
		self.single_phase_col_prop_df_dict = dict()

	def _get_overlap(self, curr_im_data, next_im_data):
		'''
		Makes overlap df of colonies between two images
		Identifies pairs of colonies between images for which the
		centroid of one falls inside the other's bounding box
		'''
		# get matrices of absolute differences between x and y
		# centroids of the colonies in the two images
		cX_diff_mat = np.abs(np.subtract.outer(
			curr_im_data['cX'].to_numpy(),
			next_im_data['cX'].to_numpy()))
		cY_diff_mat = np.abs(np.subtract.outer(
			curr_im_data['cY'].to_numpy(),
			next_im_data['cY'].to_numpy()))
		# calculate whether the other image's centroid falls
		# within each respective image's bounding box bounds
		curr_im_cX_bbox_overlap = (cX_diff_mat.T <
			curr_im_data['bb_width'].to_numpy()/2).T
		curr_im_cY_bbox_overlap = (cY_diff_mat.T <
			curr_im_data['bb_height'].to_numpy()/2).T
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
			index = curr_im_data.index,
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
		# set self.tracking_col_name to NaN for merged colonies in
		# self.active_col_prop_df
		self.active_col_prop_df.loc[
			merged_colonies, self.tracking_col_name] = np.nan
		# return a version of match_df that is missing merged colonies
		match_df_nonmerging = match_df[~colonies_that_merge_bool]
		# set self._merged_colonies_removed to True
		self._merged_colonies_removed = True
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
			split_colony_match_list.append(match_to_keep)
		# combine selected split colonies with
		# match_df_no_splits into single df
		match_df_filtered = \
			pd.concat(split_colony_match_list, sort = False)
		return(match_df_filtered)

	def _track_single_im_pair(self, curr_im_data, next_im_data):
		'''
		Performs full tracking between curr_im_data and next_im_data,
		and records results in self.active_col_prop_df
		'''
		# identify all matches between colonies in the two images
		match_df = self._id_colony_match(curr_im_data, next_im_data)
		# set self.tracking_col_name in merged colonies to NaN, and get
		# back match_df without colonies that merge into others
		match_df_nonmerging = self._tag_merged_colonies(match_df)
		# for colonies that break into multiple pieces, track the
		# largest piece as the main colony
		match_df_filtered = self._resolve_splits(match_df_nonmerging)
		# for colonies with a direct match between subsequent
		# timepoints, set self.tracking_col_name in next_timepoint to the
		# self.tracking_col_name from current_timepoint
		self.active_col_prop_df.loc[
			match_df_filtered.next_im_colony, self.tracking_col_name] = \
			self.active_col_prop_df.loc[
				match_df_filtered.curr_im_colony,
					self.tracking_col_name].values

	def match_and_track_across_time(self, phase_num, colony_prop_df):
		'''
		Identifies matching colonies between subsequent timepoints
		Performs equivalent role to ColonyAreas_mod in original matlab
		code, but see comments on class for differences
		'''
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
		self.active_col_prop_df[self.tracking_col_name] = \
			["phase_{}_xy{}_col{}".format(phase_num, xy_pos, col_id) for \
				phase_num, xy_pos, col_id in \
				zip(self.active_col_prop_df.phase_num,
					self.active_col_prop_df.xy_pos_idx,
					self.active_col_prop_df.index)]
		# convert time tracking id column to categorical
		self.active_col_prop_df[self.tracking_col_name] = \
			self.active_col_prop_df[self.tracking_col_name].astype(
				'category')
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
		# add data for this phase to dict
		self.single_phase_col_prop_df_dict[phase_num] = \
			self.active_col_prop_df.copy()
		return(self.active_col_prop_df)

	def match_and_track_across_phases(self):
		'''
		Identifies matching colonies between last timepoint of each
		phase and the first timepoint of the subsequent phase
		Performs equivalent role to ColonyAreas_mod in original matlab
		code, but see comments on class for differences
		'''
		### !!! NEEDS UNITTEST
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
				np.sort(np.array(
					self.single_phase_col_prop_df_dict.keys()))
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
			# convert cross-phase tracking id column to categorical
			self.active_col_prop_df[self.tracking_col_name] = \
				self.active_col_prop_df[self.tracking_col_name].astype(
					'category')
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
					next_phase_first_tp = np.min(curr_phase_data.timepoint)
					# loop through xy positions
					for curr_xy_pos in self._xy_positions:
						# get colony properties at current timepoint
						curr_im_data = curr_phase_data[
							(curr_phase_data.timepoint == curr_phase_last_tp) &
							(curr_phase_data.xy_pos_idx == curr_xy_pos)]
						# get colony properties at next timepoint
						next_im_data = next_phase_data[
							(next_phase_data.timepoint == next_phase_first_tp) &
							(next_phase_data.xy_pos_idx == curr_xy_pos)]
						self._track_single_im_pair(curr_im_data, next_im_data)
			self.cross_phase_tracked_col_prop_df = \
				self.active_col_prop_df.copy()
		else:
			# columns need to be included type to save to parquet later
			self.cross_phase_tracked_col_prop_df = \
				pd.DataFrame(columns = 
					['time_tracking_id', 'phase_num', 'xy_pos_idx',
						'cross_phase_tracking_id'])
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

def track_colonies_single_pos(xy_pos_idx, analysis_config_obj_df = None,
	analysis_config_file = None):
	'''
	Runs image analysis and colony tracking for xy_pos_idx
	Takes either analysis_config_obj_df or analysis_config_file as arg
	'''
	# check that only analysis_config_obj_df or
	# analysis_config_file is supplied
	if (analysis_config_obj_df is None) == (analysis_config_file is None):
		raise ValueError(
			'Must supply EITHER analysis_config_obj_df OR ' +
			'analysis_config_file argument')
	if analysis_config_obj_df is None:
		analysis_config_obj_df = \
			analysis_configuration.set_up_analysis_config(analysis_config_file)
	# set up colony tracker
	colony_tracker = ColonyTracker()
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
				phase_num, untracked_phase_pos_data)
	# track colonies across phases
	time_and_phase_tracked_pos_data = \
		colony_tracker.match_and_track_across_phases()
	# write phase-tracked file to parquet format
	time_and_phase_tracked_pos_data.to_parquet(
		analysis_config.tracked_properties_write_path)
	return(time_and_phase_tracked_pos_data)



