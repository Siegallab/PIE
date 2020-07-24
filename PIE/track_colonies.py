#!/usr/bin/python

'''
Tracks colonies through time in a single imaging field
'''

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

	def _write_position_phase_data_binary(self, phase_colony_data_tracked):
		'''
		Writes positions phase data to parquet file
		'''
		output_file = \
			os.path.join(
				self.analysis_config.phase_col_properties_output_folder,
				('xy_' + str(self.analysis_config.xy_position_idx) + 
					'_tracked_phase_data.parquet'))
		phase_colony_data_tracked.to_parquet(output_file)

	def analyze_phase_data(self, write_phase_data = False):
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
			if write_phase_data:
				self._write_position_phase_data_binary(
					phase_colony_data_tracked)
		return(phase_colony_data_tracked)

class _ColonyTracker(object):
	'''
	Generic class for tracking colonies based on dataframe of colony
	properties in an image and the subsequent image
	The details of the tracking differ from the original matlab code
	when it comes to colony splitting/merging
	In addition, dealing with filters by min_growth_time, settle_frames,
	etc is outsourced to growth rate functions
	'''
	# TODO: CURRENTLY DOESN'T HANDLE SATELLITES!
	#	need to check how those are *really* dealt with in original code
	#	it kind of seems like if they merge with main colony, it counts
	#	as a merge?
	def __init__(self, colony_prop_df):
		self.colony_prop_df = colony_prop_df
		# reset indices in self.colony_prop_df to be consecutive
		# integers, to make sure each row has a unique index that can be
		# used as an identifier
		self.colony_prop_df.reset_index(inplace = True, drop=True)
		self._init_tracking_col()
		# check that tracking column correctly initalized
		if not self.tracking_col_name in self.colony_prop_df or \
			self.colony_prop_df[self.tracking_col_name].isnull().values.all():
			raise ValueError('Tracking column ' + self.tracking_col_name + \
				' not correctly initalized')

	def _init_tracking_col(self):
		'''
		Initializes the tracking column by assigning a unique tracking
		ID to each colony in self.colony_prop_df
		'''
		pass

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
		# self.colony_prop_df
		self.colony_prop_df.loc[
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
			current_match_areas = self.colony_prop_df.loc[
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
		and records results in self.colony_prop_df
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
		self.colony_prop_df.loc[
			match_df_filtered.next_im_colony, self.tracking_col_name] = \
			self.colony_prop_df.loc[
				match_df_filtered.curr_im_colony,
					self.tracking_col_name].values

class _TimeTracker(_ColonyTracker):
	'''
	Tracks colonies through time based on dataframe of colony properties
	at each timepoint
	The details of the tracking differ from the original matlab code
	when it comes to colony splitting/merging
	In addition, dealing with filters by min_growth_time, settle_frames,
	etc is outsourced to growth rate functions
	'''
	# TODO: CURRENTLY DOESN'T HANDLE SATELLITES!
	#	need to check how those are *really* dealt with in original code
	#	it kind of seems like if they merge with main colony, it counts
	#	as a merge?

	def __init__(self, timecourse_colony_prop_df):
		self._timepoints = \
			np.sort(np.unique(timecourse_colony_prop_df.timepoint))
		self.tracking_col_name = 'time_tracking_id'
		super(_TimeTracker, self).__init__(timecourse_colony_prop_df)

	def _init_tracking_col(self):
		'''
		Initializes the tracking column by assigning a unique tracking
		ID to each colony in self.colony_prop_df
		'''
		self.colony_prop_df[self.tracking_col_name] = \
			["phase_{}_xy{}_col{}".format(phase, xy_pos, col_id) for \
				phase, xy_pos, col_id in \
				zip(self.colony_prop_df.phase,
					self.colony_prop_df.xy_pos_idx,
					self.colony_prop_df.index)]

	def match_and_track(self):
		'''
		Identifies matching colonies between subsequent timepoints
		Performs equivalent role to ColonyAreas_mod in original matlab
		code, but see comments on class for differences
		'''
		### !!! NEEDS UNITTEST
		# loop through timepoints
		for curr_timepoint, next_timepoint in \
			zip(self._timepoints[:-1], self._timepoints[1:]):
			# get df of which colony in curr_timepoint matches which
			# colony in next_timepoint
			###
			# get colony properties at current timepoint
			curr_im_data = \
				self.colony_prop_df[
					self.colony_prop_df.timepoint == curr_timepoint]
			# get colony properties at next timepoint
			next_im_data = \
				self.colony_prop_df[
					self.colony_prop_df.timepoint == next_timepoint]
			self._track_single_im_pair(curr_im_data, next_im_data)
		return(self.colony_prop_df)

def track_single_phase_single_position(xy_pos, analysis_config,
	postphase_analysis_config, write_phase_data = False):
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
		single_phase_single_pos_compiler.analyze_phase_data(write_phase_data)
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


