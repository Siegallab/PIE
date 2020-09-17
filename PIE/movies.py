#!/usr/bin/python

'''
Makes movies of colony growth
'''

import cv2
import numpy as np
import os
import pandas as pd
import plotnine as p9
import random
import warnings
from mizani.palettes import hue_pal # installed with plotnine
from io import BytesIO
from PIL import ImageColor, Image
from PIE import process_setup_file
from PIE.image_properties import create_color_overlay
from PIE.growth_measurement import get_colony_properties
from PIE.ported_matlab import bwperim

### !!! NEEDS UNITTESTS FOR THE WHOLE THING

### TODO: 	Include option to overlay fluorescence
####		add text
####		deal with postphase
####		im_label in im_df not used (can im_df just be dict?)
####		make default dict of component
####		By default colors should be picked separately within each xy position
####		need warning if non-unique colonies in list
####		need way to blend fluor channels
####		change channel to use channel name from df, not file (since file channel names can switch between phases)

class _PlotMovieMaker(object):
	'''
	Plots required timepoints based on data in col_prop_df

	If facet_override is a bool, it overrides self._facet_phase_default
	w.r.t. whether plot should be faceted by phase
	'''
	def __init__(self, plot_prop, col_prop_df, analysis_config_obj_df,
		width, height,
		facet_override = None, y_label_override = None):
		self.plot_prop = plot_prop
		self.width = width
		self.height = height
		self.analysis_config_obj_df = analysis_config_obj_df
		# add first timepoint time to col_prop_df
		self.col_prop_df = self._add_first_timepoint_data(col_prop_df)
		# set up output_im_holder
		self.output_im_holder = OutputIMHolder(self.col_prop_df)
		# add time in hours to self.col_prop_df
		self.col_prop_df['time_in_hours'] = \
			(self.col_prop_df['time_in_seconds'] - \
				self.col_prop_df['first_timepoint_time']).astype(float)/3600
		# set self.gr_df to None until it is required
		self.gr_df = None
		# determine default for whether to separate plot facets by phase
		self._determine_phase_plotting(facet_override)
		if y_label_override is None:
			self.y_label = self.plot_prop
		else:
			self.y_label = y_label_override

	def _add_first_timepoint_data(self, col_prop_df):
		'''
		Add data for the first timepoint for which data is collected
		'''
		self.first_tp_df = \
			pd.DataFrame(
				{'first_timepoint_time': np.nan},
				index = pd.Index(
					self.analysis_config_obj_df.index, name = 'phase_num'
					)
				)
		for phase in self.analysis_config_obj_df.index:
			analysis_config = \
				self.analysis_config_obj_df.at[phase, 'analysis_config']
			self.first_tp_df.at[phase, 'first_timepoint_time'] = \
				analysis_config.first_timepoint_time
		self.first_tp_df.reset_index(inplace = True)
		col_prop_df = pd.merge(col_prop_df, self.first_tp_df)
		return(col_prop_df)

	def _determine_phase_plotting(self, facet_override):
		'''
		Determines whether the data should be faceted by phase (if
		recorded phase times appear independent of each other), or
		whether every phase should be plotted on a common plot (if
		timing of each phase appears to end after the previous phase
		has ended)
		'''
		self.timing_df = \
			self.col_prop_df[
				['global_timepoint', 'phase_num', 'timepoint', 'time_in_hours']
				].drop_duplicates()
		if facet_override is None:
			# if the time constantly increases when global_timepoint is
			# ordered, then don't facet by phase
			ordered_times = \
				self.timing_df.sort_values(by = 'global_timepoint').\
					time_in_hours.to_numpy()
			time_diffs = np.diff(ordered_times)
			if np.all(time_diffs > 0):
				self._facet_phase = False
			else:
				self._facet_phase = True
		else:
			if not isinstance(facet_override, bool):
				raise TypeError('facet_override must be None or a bool')
			self._facet_phase = facet_override

	def _plot_property(self, last_glob_tp):
		'''
		Plots self.plot_prop over time, ending with last_glob_tp
		(but on axis that is invariable to last_glob_tp)

		If property_label is not None, y axis is relabeled with
		property_label
		'''
		# convert phase_num to string for plotting
		mod_col_prop = self.col_prop_df.copy()
		mod_col_prop.phase_num = mod_col_prop.phase_num.astype(str)
		df_to_plot = \
			self.col_prop_df[
				self.col_prop_df.global_timepoint <= last_glob_tp
				].copy()
		df_to_plot.phase_num = df_to_plot.phase_num.astype(str)
		# pass full data to ggplot for scales
		ggplot_obj = \
			p9.ggplot(data = mod_col_prop) + \
			p9.geom_point(
				data = df_to_plot,
				mapping = p9.aes(
					x = 'time_in_hours',
					y = self.plot_prop,
					color = 'hex_color',
					shape = 'phase_num'),
				size = 2,
				) + \
			p9.scale_colour_identity() + \
			p9.guides(shape = self._facet_phase) + \
			p9.theme(legend_position="bottom",
				plot_margin = 0) + \
			p9.theme_bw()
		# determine whether to facet by on phase_num
		if self._facet_phase:
			ggplot_obj = ggplot_obj + \
				p9.facet_wrap('~phase_num')
		# add x and y axis properties
		ggplot_obj = ggplot_obj + \
			p9.scale_x_continuous(
				name = 'time (hours)',
				limits = (
					self.col_prop_df['time_in_hours'].min()-0.5,
					self.col_prop_df['time_in_hours'].max()+0.5
					)
				) + \
			p9.scale_y_continuous(
				name = self.y_label,
				limits = (
					self.col_prop_df[self.plot_prop].min()/1.1,
					self.col_prop_df[self.plot_prop].max()*1.1
					)
				)
		return(ggplot_obj)

	def _plot_to_np(self, ggplot_obj):
		'''
		Returns rgb numpy array of ggplot_obj plot

		Warning: height and width are not exactly self.height and
		self.width (consistent with ggplot issues)
		'''
		# modified from https://stackoverflow.com/a/58641662/8082611
		buf = BytesIO()
		# create fake width and height based on pixels
		fake_dpi = 300 # this parameter doesn't matter at all
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			ggplot_obj.save(
				filename=buf,
				format='png',
				width=self.width/fake_dpi,
				height=self.height/fake_dpi,
				units='in', dpi=fake_dpi,
				limitsize=False)
		buf.seek(0)
		img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
		buf.close()
		img = cv2.imdecode(img_arr, 1)
#		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img

	def _create_plot(self, last_glob_tp):
		'''
		Creates plot as image, plotting all points up to and including
		last_glob_tp
		'''
		ggplot_obj = self._plot_property(last_glob_tp)
		plot_im = self._plot_to_np(ggplot_obj)
		return(plot_im)

	def generate_movie_ims(self):
		for global_tp in self.col_prop_df.global_timepoint:
			plot_im = self._create_plot(global_tp)
			self.output_im_holder.add_im(global_tp, plot_im, '')
		return(self.output_im_holder.im_df)

class _GrowthPlotMovieMaker(_PlotMovieMaker):
	'''
	Plots growth across required timepoints based on data in col_prop_df

	If facet_override is a bool, it overrides self._facet_phase_default
	w.r.t. whether plot should be faceted by phase
	'''

	def __init__(self, col_prop_df, analysis_config_obj_df,
		width, height, color_df,
		facet_override = None, add_growth_line = True, add_lag_line = True):
		self.add_growth_line = add_growth_line
		self.add_lag_line = add_lag_line
		self.color_df = color_df
		super(_GrowthPlotMovieMaker, self).__init__(
			'ln_area', col_prop_df, analysis_config_obj_df,
			width, height, facet_override, y_label_override = 'ln(area)')

	def _set_start_end_times_and_log_areas(self, gr_df):
		'''
		Set times for starting and ending growth tracking in gr_df
		'''
		# use timings from self.col_prop_df to set time_in_hrs  of t0
		# and tfinal
		key_df_time = self.timing_df[['timepoint', 'time_in_hours']]
		t0_key = key_df_time.rename(
			columns={
				'timepoint': 't0',
				'time_in_hours': 't0_in_hours'})
		tfinal_key = key_df_time.rename(
			columns={'timepoint': 'tfinal',
				"time_in_hours": 'tfinal_in_hours'})
		gr_df = pd.merge(left = gr_df, right = t0_key)
		gr_df = pd.merge(left = gr_df, right = tfinal_key)
#		# use ln_area from self.col_prop_df to set ln_area of t0 and
#		# tfinal
#		key_df_area = self.col_prop_df[
#			['cross_phase_tracking_id', 'ln_area', 'timepoint', 'phase_num']
#			]
#		t0_area_key = key_df_area.rename(
#			columns={
#				'timepoint': 't0',
#				'ln_area': 'ln_area_t0'})
#		tfinal_area_key = key_df_area.rename(
#			columns={
#				'timepoint': 'tfinal',
#				'ln_area': 'ln_area_tfinal'})
#		gr_df = pd.merge(left = gr_df, right = t0_area_key) 
#		gr_df = pd.merge(left = gr_df, right = tfinal_area_key)
		gr_df['yhat_t0'] = gr_df.t0_in_hours * gr_df.gr + gr_df.intercept
		gr_df['yhat_tfinal'] = gr_df.tfinal_in_hours * gr_df.gr + gr_df.intercept
		return(gr_df)

	def _get_gr_data(self):
		'''
		Reads in growth rate data from each phase
		
		Subsets data to only include colonies with
		cross_phase_tracking_id from self.col_prop_df

		Adds color data
		'''
		gr_data_list = []
		for analysis_config in self.analysis_config_obj_df.analysis_config:
			current_data = pd.read_csv(analysis_config.phase_gr_write_path)
			gr_data_list.append(current_data)
		gr_data = pd.concat(gr_data_list, sort = False).reset_index(
			drop = True)
		unique_ids = self.col_prop_df.cross_phase_tracking_id.unique()
		# subset relevant data
		gr_data_subset = \
			gr_data.loc[gr_data.cross_phase_tracking_id.isin(unique_ids)]
		# add colors
		gr_df_colored = pd.merge(left = gr_data_subset, right = self.color_df)
		# add t0 and tfinal times and areas
		gr_df_extra_times = self._set_start_end_times_and_log_areas(gr_df_colored)
		# change phase_num to string type
		gr_df_extra_times.phase_num = gr_df_extra_times.phase_num.astype(str)
		self.gr_df = gr_df_extra_times

	def _plot_growth_rate(self, last_glob_tp):
		'''
		Plots log area over time like self.plot_prop

		Optionally adds growth and lag lines, if growth rate data has
		been calculated
		'''
		# add ln_area column for growth rate plotting
		if 'ln_area' not in self.col_prop_df.columns:
			self.col_prop_df['ln_area'] = np.log(self.col_prop_df.area)
		# plot ln_area points
		ggplot_obj = self._plot_property(last_glob_tp)
		# only add growth and lag line at last timepoint
		if int(last_glob_tp) == \
			self.col_prop_df.global_timepoint.astype(int).max():
			# need growth rate data only for plotting growth and lag lines
			if self.add_growth_line or self.add_lag_line:
				# read in growth rate data
				self._get_gr_data()
			if self.add_growth_line:
				ggplot_obj = ggplot_obj + \
					p9.geom_segment(
						data = self.gr_df,
						mapping = p9.aes(
							x = 't0_in_hours',
							y = 'yhat_t0',
							xend = 'tfinal_in_hours',
							yend = 'yhat_tfinal',
							color = 'hex_color'),
						size = 1)
			if self.add_lag_line:
				ggplot_obj = ggplot_obj + \
					p9.geom_segment(
						data = self.gr_df,
						mapping = p9.aes(
							x = 0,
							y = 'yhat_t0',
							xend = 'lag',
							yend = 'yhat_t0',
							color = 'hex_color'),
						size = 1,
						linetype = 'dashed')
		return(ggplot_obj)

	def _create_plot(self, last_glob_tp):
		'''
		Creates plot as image, plotting all points up to and including
		last_glob_tp
		'''
		ggplot_obj = self._plot_growth_rate(last_glob_tp)
		plot_im = self._plot_to_np(ggplot_obj)
		return(plot_im)

class _PositionMovieMaker(object):
	'''
	Makes movies of colonies in col_prop_df across time/phases
	Colonies in col_prop_df must represent a single xy position

	expansion_pixels is the number of pixels outside of the outermost
	colonies (on each side) to include in the movie

	col_shading_alpha is the opacity of the coloring over each colony in
	movies (if 0, only bounds shown; if 1, colony completely covered in
	color overlay)

	bound_width is the width of the colored colony boundaries in movies
	(if 0, no bounds shown; otherwise, width is in pixels, but on the
	scale of the original image)

	base_fluor_channel is the fluorescent channel name to use for images;
	if None (default) uses main_channel_label

	normalize_intensity is a bool that determines whether input image
	should be normalized; if True, the image bitdepth must be passed

	bitdepth is an optional argument for the bitdepth of the input image
	'''
	def __init__(self, analysis_config_obj_df, col_prop_df,
		col_shading_alpha, bound_width, normalize_intensity,
		expansion_pixels, base_fluor_channel = None, bitdepth = None,
		postphase_fluor_channel = None):
		self.col_prop_df = col_prop_df
		self.xy_pos_idx = self._get_single_val_from_df('xy_pos_idx', col_prop_df)
		self.analysis_config_obj_df = analysis_config_obj_df
		self.normalize_intensity = bool(normalize_intensity)
		self.base_fluor_channel = base_fluor_channel
		self.postphase_fluor_channel = postphase_fluor_channel
		self.expansion_pixels = int(expansion_pixels)
		self.col_shading_alpha = col_shading_alpha
		self.bound_width = bound_width
		self.bitdepth = bitdepth
		# check data
		self._perform_data_checks()
		# identify bounds of movie in image
		# (assume all images are the same size)
		self._id_im_bounds()
		# initialize output im holder object
		self.output_im_holder = OutputIMHolder(col_prop_df)

	def _get_single_val_from_df(self, param, df):
		'''
		Gets what should be a single value from column 'param' in df
		Throws error if value not unique
		'''
		vals = df[param].unique()
		if len(vals) > 1:
			raise ValueError('More than one ' + param +
				'identified in df \n' +
				str(df))
		else:
			val = vals[0]
		return(val)

	def _perform_data_checks(self):
		if self.col_shading_alpha < 0 or self.col_shading_alpha > 1:
			raise ValueError('col_shading_alpha must be between 0 and 1')
		if self.bound_width < 0:
			raise ValueError('bound_width must be 0 or higher')
		if not self.normalize_intensity and not isinstance(self.bitdepth, int):
			raise ValueError('If normalize_intensity is set to False, input '
				'image bitdepth must be passed as an integer')

	def _id_im_bounds(self):
		'''
		Identifies bounding pixels of movie
		'''
		temp_analysis_config = self.analysis_config_obj_df.iloc[0]['analysis_config']
		self.y_start = np.max([
			0, np.min(self.col_prop_df['bb_y_top'].astype(int) - self.expansion_pixels)
			])
		self.x_start = np.max([
			0, np.min(self.col_prop_df['bb_x_left'].astype(int) - self.expansion_pixels)
			])
		self.y_range_end = np.min([
			temp_analysis_config.im_height,
			np.max((self.col_prop_df['bb_y_top'] + 
				self.col_prop_df['bb_height']).astype(int) + self.expansion_pixels + 1)
			])
		self.x_range_end = np.min([
			temp_analysis_config.im_width,
			np.max((self.col_prop_df['bb_x_left'] + 
				self.col_prop_df['bb_width']).astype(int) + self.expansion_pixels + 1)
			])
		self.im_width = self.x_range_end - self.x_start
		self.im_height = self.y_range_end - self.y_start

	def _color_colony(self, col_row, input_im, labeled_mask,
		override_shading = False):
		'''
		Adds overlay (with opacity dependent on self.col_shading_alpha)
		and boundary (with width dependent on
		self.bound_width) to a single colony
		'''
		colony_mask = labeled_mask == col_row.label
		if override_shading or self.col_shading_alpha == 0:
			shaded_im = input_im
		else:
			# draw overlay for current colony on input_im
			shaded_im = create_color_overlay(input_im, colony_mask,
				col_row.rgb_color, self.col_shading_alpha)
		# draw boundary of current colony on input_im
		if self.bound_width > 0:
			colony_boundary_mask = \
				np.zeros(colony_mask.shape, dtype = bool)
			for bound_counter in range(0,self.bound_width):
				# boundary is previous boundary combined with outline of
				# previous colony_mask
				colony_boundary_mask = np.logical_or(colony_boundary_mask,
					bwperim(colony_mask))
				# next colony_mask is previous colony_mask minus current
				# boundary
				colony_mask = \
					np.logical_and(np.invert(colony_boundary_mask),
						colony_mask)
			output_im = create_color_overlay(shaded_im, colony_boundary_mask,
				col_row.rgb_color, 1)
		else:
			output_im = shaded_im
		return(output_im)

	def _create_subframe(self, frame):
		'''
		Crops frame based on values created in self._id_im_bounds
		'''
		subframe = \
			frame[self.y_start:self.y_range_end, self.x_start:self.x_range_end]
		return(subframe)

	def _generate_frame(self, tp_col_prop_df, analysis_config, timepoint,
		channel_label):
		'''
		Generates a boundary image for global timepoint global_tp
		'''
		global_tp = \
			self._get_single_val_from_df('global_timepoint', tp_col_prop_df)
		# get labeled colony mask and input file
		mask_image_name = analysis_config.create_file_label(
			timepoint,
			analysis_config.xy_position_idx,
			analysis_config.main_channel_label)
		colony_mask_path = \
			os.path.join(
				analysis_config.phase_output_path,
				'colony_masks',
				mask_image_name + '.tif'
				)
		labeled_mask_full = cv2.imread(colony_mask_path, cv2.IMREAD_ANYDEPTH)
		input_im_full, im_name, _ = \
			analysis_config.get_image(timepoint, channel_label)
		if input_im_full is None:
			raise ValueError(
				'Missing input im for timepoint '+str(timepoint)+
				'and channel ' + channel)
		# subframe mask and input im
		input_im = self._create_subframe(input_im_full)
		labeled_mask = self._create_subframe(labeled_mask_full)
		# convert input_im to 8-bit, normalizing if necessary
		if self.normalize_intensity:
			im_8_bit = cv2.normalize(input_im, None, alpha=0, beta=(2**8-1),
				norm_type=cv2.NORM_MINMAX)
		else:
			# change bitdepth to 8-bit
			im_8_bit = \
				np.uint8(np.round(input_im.astype(float)/(2**(self.bitdepth - 8))))
		# initialize image with overlay
		im_with_overlay = cv2.cvtColor(im_8_bit, cv2.COLOR_GRAY2RGB)
		# loop over colonies in tp_col_prop_df, add colored overlay for
		# each one
		for col_row in tp_col_prop_df.itertuples():
			im_with_overlay = \
				self._color_colony(col_row, im_with_overlay, labeled_mask)
		# add image with overlay to output_im_holder
		self.output_im_holder.add_im(global_tp, im_with_overlay, im_name)

	def _generate_postphase_frame(self, phase_col_prop_df, analysis_config,
		postphase_analysis_config, channel_label):
		'''
		Generates a boundary image for postphase fluorescent image
		'''
		# get timepoint at which to draw boundary for each colony
		timepoint_label = \
			postphase_analysis_config.fluor_channel_df[
				postphase_analysis_config.fluor_channel_df.fluor_channel_label==
					channel_label]['fluor_timepoint'][0]
		# need to import label colony property mat dataframe
		label_property_mat_path = \
			analysis_config.get_property_mat_path('label')
		label_property_mat_df = pd.read_csv(label_property_mat_path, index_col = 0)
		# make df that holds label, timepoint to use, and
		# cross_phase_tracking_id for each colony
		colony_df = phase_col_prop_df[
			['time_tracking_id', 'cross_phase_tracking_id', 'rgb_color']
			].drop_duplicates()
		# read in growth rate df if necessary
		if isinstance(timepoint_label, str) and \
			timepoint_label in ['first_gr', 'last_gr']:
			gr_df = pd.read_csv(analysis_config.phase_gr_write_path, index_col = 0)
			indices_to_get = list(
				set(gr_df.index).intersection(
					set(colony_df.time_tracking_id.to_list())))
			colony_df = colony_df[colony_df.time_tracking_id.isin(indices_to_get)]
		else:
			gr_df = None
		# get timepoint labels from current phase corresponding to self.
		colony_df['label'], colony_df['timepoint'] = \
			get_colony_properties(
				label_property_mat_df,
				timepoint_label,
				index_names = colony_df.time_tracking_id.to_list(),
				gr_df = gr_df
				)
		# get input image for postphase fluorescent channel
		input_im_full, im_name, _ = \
			postphase_analysis_config.get_image(None, channel_label)
		# subframe input im
		input_im = self._create_subframe(input_im_full)
		# convert input_im to 8-bit, normalizing if necessary
		if self.normalize_intensity:
			im_8_bit = cv2.normalize(input_im, None, alpha=0, beta=(2**8-1),
				norm_type=cv2.NORM_MINMAX)
		else:
			# change bitdepth to 8-bit
			im_8_bit = \
				np.uint8(np.round(input_im.astype(float)/(2**(self.bitdepth - 8))))
		# initialize image with overlay
		im_with_overlay = cv2.cvtColor(im_8_bit, cv2.COLOR_GRAY2RGB)
		# loop through timepoints in colony_df and draw bounds on
		# colonies corresponding to each
		for timepoint in colony_df.timepoint.unique():
			# get part of colony_df corresponding to the current
			# timepoint
			tp_col_prop_df = colony_df[colony_df.timepoint == timepoint]
			# get labeled colony mask and input file
			mask_image_name = analysis_config.create_file_label(
				int(timepoint),
				analysis_config.xy_position_idx,
				analysis_config.main_channel_label)
			colony_mask_path = \
				os.path.join(
					analysis_config.phase_output_path,
					'colony_masks',
					mask_image_name + '.tif'
					)
			labeled_mask_full = cv2.imread(colony_mask_path, cv2.IMREAD_ANYDEPTH)
			if input_im_full is None:
				raise ValueError(
					'Missing input im for timepoint '+str(timepoint)+
					'and channel ' + channel)
			# subframe mask
			labeled_mask = self._create_subframe(labeled_mask_full)
			# loop over colonies in tp_col_prop_df, add colored overlay for
			# each one
			for col_row in tp_col_prop_df.itertuples():
				im_with_overlay = \
					self._color_colony(col_row, im_with_overlay, labeled_mask,
						override_shading = True)
		# add image with overlay to output_im_holder
		global_tp = phase_col_prop_df.global_timepoint.max() + 0.1
		self.output_im_holder.add_im(global_tp, im_with_overlay, im_name)

	def generate_movie_ims(self):
		for phase in self.analysis_config_obj_df.index:
			analysis_config = \
				self.analysis_config_obj_df.at[phase, 'analysis_config']
			analysis_config.set_xy_position(self.xy_pos_idx)
			postphase_analysis_config = \
				self.analysis_config_obj_df.at[phase,
					'postphase_analysis_config']
			# get channel label to use for main image analysis
			if self.base_fluor_channel is None:
				channel_label = analysis_config.main_channel_label
			else:
				channel_label = self.base_fluor_channel
			phase_col_prop_df = self.col_prop_df[
				self.col_prop_df.phase_num == phase]
			for timepoint in phase_col_prop_df.timepoint.unique():
				tp_col_prop_df = \
					phase_col_prop_df[phase_col_prop_df.timepoint == 
						timepoint]
				self._generate_frame(tp_col_prop_df, analysis_config,
					int(timepoint), channel_label)
			# get postphase image if necessary
			if postphase_analysis_config != None and \
				self.postphase_fluor_channel != None:
				postphase_analysis_config.set_xy_position(self.xy_pos_idx)
				self._generate_postphase_frame(
					phase_col_prop_df,
					analysis_config,
					postphase_analysis_config,
					self.postphase_fluor_channel)
		return(self.output_im_holder.im_df)

class OutputIMHolder(object):

	def __init__(self, col_prop_df):
		'''
		Generate an empty dataframe for holding output images
		'''
		self.im_df = pd.DataFrame(
			columns = ['global_timepoint', 'im', 'im_name'])
		self.im_df.global_timepoint = \
			np.sort(col_prop_df.global_timepoint.unique())
		self.im_df.set_index('global_timepoint', inplace = True)

	def add_im(self, global_timepoint, im, im_name):
		self.im_df.at[global_timepoint, 'im'] = im
		self.im_df.at[global_timepoint, 'im_name'] = im_name

class _SubframeCombiner(object):
	'''
	Concatenates parts of movie images (e.g. cell movie, plots, etc)
	into single set of movie images

	im_width is the width of the final movie images

	im_height is the height of the final movie images

	ordered_component_string is a string of movie labels with commas
	separating components in the same row of the movie frame and
	semicolons separating rows, and blank representing empty spaces
	(e.g. 'bf_movie | blank; rfp_movie|gfp_movie; growth_plot' has 3
	rows, with bf_movie in the upper left and nothing in the upper
	right, rfp_movie and gfp_movie in the second row, and growth_plot
	taking up the entire bottom row

	component_size_dict is a dict with keys as labeled movie components
	and values being (w, h) tuples for the size of each component
	row and column values must add up to im_height and im_width,
	respectively, although blank spacers will be used to take up any
	blank space (multiple blank values per row, or multiple instances of
	blank values taking up entire rows, will be sized equally in the x-
	or y-direction, respectively)

	blank_color is the hex value for the color of empty spaces between
	movies (default is white, #FFFFFF)

	'''
	def __init__(self, im_width, im_height, ordered_components,
		component_size_dict, blank_color = '#FFFFFF'):
		self.im_height = im_height
		self.im_width = im_width
		self.blank_color = blank_color
		self.component_size_dict = component_size_dict
		self._parse_subframe_positions(ordered_components)

	def _set_x_offsets_subframe_row(self, subframe_row):
		'''
		Checks that elements in subframe row add up in size to
		self.im_width, and are the same height

		Populates x offsets for non-blank elements of subframe_row in
		self._offset_df

		Returns column height
		'''
		blank_number = subframe_row.count('blank')
		real_elements = [x for x in subframe_row if x != 'blank']
		# check that real_elements all have associated sizes
		if not set(real_elements).issubset(
			set(self.component_size_dict.keys())):
			raise ValueError(
				'row ' + str(subframe_row) +
				'contains elements not found in keys of compnent_size_dict: ' +
				str(self.component_size_dict.keys()))
		real_element_num = len(real_elements)
		if real_element_num > 0:
			real_element_widths = \
				[self.component_size_dict[x][0] for x in real_elements]
			real_element_heights = \
				[self.component_size_dict[x][1] for x in real_elements]
			# assume column height is the height of the first real
			# element and then check
			column_height = real_element_heights[0]
			# check that column_height appears in every position in
			# real_element_heights
			if real_element_heights.count(column_height) != real_element_num:
				raise ValueError(
					'Different heights passed for ' + str(real_elements) +
					' although they are listed in the same row'
					)
			# check element widths and calculate size of any blanks
			combined_element_width = sum(real_element_widths)
			width_diff = self.im_width - combined_element_width
			if width_diff < 0:
				raise ValueError(
					'Combined width of row ' + str(subframe_row) +
					' is ' + str(combined_element_width) +
					', which exceeds the width of the full movie image (' +
					str(self.im_width) + ')'
					)
			elif width_diff > 0:
				if blank_number == 0:
					raise ValueError(
						'Combined width of row ' +str(subframe_row) +
						' is ' + str(combined_element_width) +
						', which is below the width of the full movie image (' +
						str(self.im_width) + '); pad row with blank elements'
						)
				else:
					blank_width = width_diff/blank_number
			else:
				blank_width = 0
				### TODO: Raise warning if blank_number > 0
			# loop through elements, setting offsets
			current_offset = 0
			for subframe in subframe_row:
				if subframe == 'blank':
					current_offset = current_offset + blank_width
				else:
					self._offset_df.at[subframe, 'x'] = current_offset
					current_offset = \
						current_offset + self.component_size_dict[subframe][0]
		return(column_height)

	def _set_y_offsets_subframe_row(self, subframe_row, y_offset):
		'''
		Populates y offsets for non-blank elements of subframe_row in
		self._offset_df
		'''
		for subframe in subframe_row:
			if subframe != 'blank':
				self._offset_df.at[subframe, 'y'] = int(y_offset)

	def _parse_subframe_positions(self, ordered_components):
		'''
		Parses order and size of components to create a df of x- and
		y-offsets for each component
		'''
		parsed_component_order = \
			[[col.strip() for col in row.split('|')]
				for row in ordered_components.split(';')]
		self._offset_df = pd.DataFrame(
			index = self.component_size_dict.keys(),
			columns = ['x', 'y'])
		# determine the number of rows containing only blanks
		blank_row_number = parsed_component_order.count(['blank'])
		if len(parsed_component_order) == blank_row_number:
			raise ValueError('No non-blank rows in ordered_components: ' +
				str(ordered_components))
		# loop through rows in ordered_components, filling in x offsets
		col_height_list = []
		for subframe_row in parsed_component_order:
			if subframe_row == ['blank']:
				col_height = 0
			else:
				col_height = self._set_x_offsets_subframe_row(subframe_row)
			col_height_list.append(col_height)
		combined_col_height = sum(col_height_list)
		height_diff = self.im_height - combined_col_height
		if height_diff < 0:
			raise ValueError(
				'Combined height of rows is ' + str(combined_col_height) +
				', which exceeds the height of the full movie image (' +
				str(self.im_height) + ')'
				)
		elif height_diff > 0:
			if blank_row_number == 0:
				raise ValueError(
					'Combined height of rows is ' + str(combined_col_height) +
					', which is below the height of the full movie image (' +
					str(self.im_height) + '); pad row with blank elements'
					)
			else:
				blank_row_height = height_diff/blank_row_number
		else:
			blank_row_height = 0
			### TODO: Raise warning if blank_number > 0
		# loop through rows, setting y offsets
		current_y_offset = 0
		for subframe_row, col_height in \
			zip(parsed_component_order, col_height_list):
			if subframe_row == ['blank']:
				current_y_offset = current_y_offset + blank_row_height
			else:
				self._set_y_offsets_subframe_row(
					subframe_row, current_y_offset
					)
				current_y_offset = current_y_offset + col_height

	def _create_blank_im(self):
		'''
		Returns blank image of correct color and size
		'''
		blank_im = \
			Image.new('RGB', (self.im_width, self.im_height), self.blank_color)
		return(blank_im)

	def _place_subframe(self, subframe_label, subframe, main_im):
		'''
		Resizes subframe and places it on main_im
		'''
		# convert subframe to Pillow Image object
		subframe_image = Image.fromarray(subframe.astype('uint8'), 'RGB')
		# resize subframe
		required_size = self.component_size_dict[subframe_label]
		if subframe_image.size != required_size:
			print(subframe_image.size)
			subframe_image = subframe_image.resize(required_size)
		# place subframe
		offset_position = tuple(self._offset_df.loc[subframe_label,['x','y']].astype(int))
		main_im.paste(subframe_image, offset_position)
		return(main_im)

	def combine_subframes(self, subframe_dict):
		'''
		Combine sets of images in subframe_dict into single list of images
		'''
		full_movie_dict = dict()
		for subframe_label, subframe_im_df in subframe_dict.items():
			# loop through global_timepoints in movie_im_df and place
			# movie image subframes
			for tp in subframe_im_df.index:
				subframe = subframe_im_df.at[tp, 'im']
				# if the current timepoint is already in
				# full_movie_dict, use that image; otherwise, create
				# new blank image
				if tp in full_movie_dict:
					main_im = full_movie_dict[tp]
				else:
					main_im = self._create_blank_im()
				# place subframe on main_im and save it in dict
				full_movie_dict[tp] = \
					self._place_subframe(subframe_label, subframe, main_im)
		return(full_movie_dict)

class MovieGenerator(object):
	'''
	Makes movies for individual aspects of colony growth (e.g. colony
	outlines, fluorescence, etc) and combines them into a single frame

	Where colonies in crossphase_colony_id_list are part of the same
	field, they will be included in the movie together

	analysis_config_file is the path to the setup file

	crossphase_colony_id_list is a list of crossphase_colony_ids for
	which movies should be generated

	movie_output_path is the location in which movies will be saved

	colony_colors is a set of HEX code strings for colors of every
	colony in crossphase_colony_id_list (in order); if None (default)
	will auto-generate colors using p9 scheme instead
	'''
	def __init__(self, analysis_config_file, crossphase_colony_id_list,
				movie_output_path, colony_colors = None):
		self.analysis_config_obj_df = \
			process_setup_file(analysis_config_file)
		# read in colony properties df
		temp_analysis_config_standin = \
			self.analysis_config_obj_df.iloc[0]['analysis_config']
		comb_colony_prop_df = \
			temp_analysis_config_standin.get_colony_data_tracked_df(
				filter_by_phase = False)
		# add 'global' cross-phase timepoint to col property df
		comb_colony_prop_df_glob_tp = \
			self._add_global_timepoints(comb_colony_prop_df)
		# add color columns to comb_colony_prop_df
		self.color_df = _generate_colony_colors(crossphase_colony_id_list,
			colony_colors)
		comb_colony_prop_df_final = \
			pd.merge(left = comb_colony_prop_df_glob_tp, right = self.color_df)
		# generate dictionary of colony properties for
		# crossphase_colony_id_list, separated by xy position
		self._subset_colony_prop_df(comb_colony_prop_df_final,
			crossphase_colony_id_list)
		# make dict of folders into which movies for each xy position
		# will be saved
		self._prep_movie_folders(movie_output_path)
		# create a df that will contain movies for every position
		self.movie_obj_df = pd.DataFrame(index = self.xy_positions)

	def _prep_movie_folders(self, movie_output_path):
		'''
		Creates a subfolder in movie_output_path for every xy position
		Creates a dictionary with xy positions and corresponding subfolders
		'''
		if not os.path.isdir(movie_output_path):
			os.makedirs(movie_output_path)
		self.movie_output_path = movie_output_path
		self.movie_subfolder_dict = dict()
		for xy_pos_idx in self.xy_positions:
			current_subfolder = \
				os.path.join(movie_output_path, 'xy_' + str(xy_pos_idx))
			self.movie_subfolder_dict[xy_pos_idx] = current_subfolder

	def _add_global_timepoints(self, comb_colony_prop_df):
		'''
		Adds a cross-phase timepoint to comb_colony_prop_df
		'''
		# get ordered list of phases
		phase_list = np.sort(self.analysis_config_obj_df.index)
		# initialize global_timepoint column
		comb_colony_prop_df['global_timepoint'] = None
		# initialize the number of timepoints to add to the next
		# phase
		prev_phases_tps = 0
		for phase in phase_list:
			# add all previous timepoints to current timepoint to make
			# global_timepoint column
			comb_colony_prop_df.loc[
				comb_colony_prop_df.phase_num == phase, 'global_timepoint'] = \
				comb_colony_prop_df.timepoint[
					comb_colony_prop_df.phase_num == phase] + prev_phases_tps
			# update the number of timepoints to add to the next phase
			prev_phases_tps = prev_phases_tps + \
				self.analysis_config_obj_df.at[phase,
					'analysis_config'].total_timepoint_num
		return(comb_colony_prop_df)

	def _subset_colony_prop_df(self, comb_colony_prop_df,
		crossphase_colony_id_list):
		'''
		Creates dict of dataframes containing only colony properties of
		colonies in crossphase_colony_id_list, separated by xy position
		'''
		col_prop_df = \
			comb_colony_prop_df[
				comb_colony_prop_df.cross_phase_tracking_id.isin(
					crossphase_colony_id_list)].copy()
		self.xy_positions = col_prop_df.xy_pos_idx.unique()
		self.xy_pos_col_prop_dict = dict()
		for xy_pos_idx in self.xy_positions:
			self.xy_pos_col_prop_dict[xy_pos_idx] = \
				col_prop_df[col_prop_df.xy_pos_idx == xy_pos_idx].copy()

	def _check_movie_label(self, movie_label):
		'''
		Checks that movie label is allowed
		'''
		if not isinstance(movie_label, str):
			raise TypeError('movie_label must be a string')
		if movie_label == 'blank':
			raise ValueError('blank is a reserved name and may not be '
				'used for movie component labels')

	def _get_ordered_image_list(self, compiled_movie_dict):
		'''
		Puts images from compiled_movie_dict into list in correct order
		'''
		ordered_timepoints = np.sort(list(compiled_movie_dict.keys()))
		images_to_save = [compiled_movie_dict[tp] for tp in ordered_timepoints]
		return(images_to_save)

	def _write_movie(self, compiled_movie_dict, output_path, duration, fourcc):
		'''
		Writes movie of fourcc format using cv2
		'''
		# make sure images saved in order
		images_to_save = self._get_ordered_image_list(compiled_movie_dict)
		images_as_cv2 = \
			[cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR)
				for img in images_to_save]
		fps = 1000.0/duration
		framesize = (images_as_cv2[0].shape[1],images_as_cv2[0].shape[0])
		# cv2 can't overwrite, so manually remove output_path if it
		# exists
		if os.path.isfile(output_path):
			os.remove(output_path)
		video = \
			cv2.VideoWriter(
				output_path,
				fourcc,
				fps,
				framesize)
		for image in images_as_cv2:
			video.write(image)
		cv2.destroyAllWindows()
		video.release()

	def _write_gif(self, compiled_movie_dict, output_path, duration, loop):
		'''
		Writes gif to output_path from images in compiled_movie_dict

		duration is duration of each frame in milliseconds

		loop is the number of times gif loops
		(0 means forever, 1 means no looping)
		'''
		# make sure images saved in order
		images_to_save = self._get_ordered_image_list(compiled_movie_dict)
		# solution to background noise from
		# https://medium.com/@Futong/how-to-build-gif-video-from-images-with-python-pillow-opencv-c3126ce89ca8
		byteframes = []
		for img in images_to_save:
		    byte = BytesIO()
		    byteframes.append(byte)
		    img.save(byte, format="GIF")
		gifs = [Image.open(byteframe) for byteframe in byteframes]
		gifs[0].save(
			output_path,
			save_all=True,
			append_images=gifs[1:],
			duration=duration,
			loop=loop)

	def _write_jpeg(self, compiled_movie_dict, output_dir, jpeg_quality):
		'''
		Saves movie as jpegs in output_dir
		'''
		for tp, im in compiled_movie_dict.items():
			filename = os.path.join(output_dir, str(tp)+'.jpg')
			im.save(filename, quality = jpeg_quality)

	def _write_tiff(self, compiled_movie_dict, output_dir):
		'''
		Saves movie as tiffs in output_dir
		'''
		for tp, im in compiled_movie_dict.items():
			filename = os.path.join(output_dir, str(tp)+'.tif')
			im.save(filename)

	def _save_movie(self, compiled_movie_dict, xy_pos_idx, movie_format,
		duration, loop, jpeg_quality):
		'''
		Saves movie for each xy_pos_idx in format specificed by
		movie_format, which can be 'jpeg'/'jpg', 'tiff'/'tif', 'gif',
		or video codecs ('h264' or 'mjpg', which both save to .mov
		format)
		'''
		movie_format = movie_format.lower()
		if movie_format in ['tiff', 'tif', 'jpeg', 'jpg']:
			output_dir = self.movie_subfolder_dict[xy_pos_idx]
			if not os.path.isdir(output_dir):
				os.makedirs(output_dir)
			if movie_format in ['tiff', 'tif']:
				self._write_tiff(compiled_movie_dict, output_dir)
			elif movie_format in ['jpg', 'jpeg']:
				self._write_jpeg(
					compiled_movie_dict, output_dir, jpeg_quality)
		elif movie_format == 'gif':
			output_path = \
				os.path.join(self.movie_output_path,
					'xy' + str(xy_pos_idx) + '.gif')
			self._write_gif(
				compiled_movie_dict, output_path, duration, loop)
		elif movie_format == 'h264':
			output_path = \
				os.path.join(self.movie_output_path,
					'xy' + str(xy_pos_idx) + '.mov')
			fourcc = cv2.VideoWriter_fourcc(*'H264')
			self._write_movie(
				compiled_movie_dict, output_path, duration, fourcc)
		elif movie_format == 'mjpeg':
			output_path = \
				os.path.join(self.movie_output_path,
					'xy' + str(xy_pos_idx) + '.mov')
			fourcc = cv2.VideoWriter_fourcc(*'MJPG')
			self._write_movie(
				compiled_movie_dict, output_path, duration, fourcc)
		else:
			raise ValueError('Unrecognized movie format ' + movie_format)

	def make_cell_movie(self, movie_label, col_shading_alpha,
		bound_width,
		normalize_intensity, expansion_pixels = 10, bitdepth = None):
		'''
		Sets up movies for main (brightfield or phase contrast) channel

		movie_label must be a string that isn't 'blank'		
		'''
		self._check_movie_label(movie_label)
		# initialize column in movie_obj_df for images for this movie
		self.movie_obj_df[movie_label] = None
		for xy_pos_idx in self.xy_positions:
			movie_maker = _PositionMovieMaker(
				self.analysis_config_obj_df,
				self.xy_pos_col_prop_dict[xy_pos_idx],
				col_shading_alpha,
				bound_width,
				normalize_intensity,
				expansion_pixels,
				bitdepth = bitdepth)
			# Keep movie_maker object in current df position
			# movies will be generated in combine_movie
			self.movie_obj_df.at[xy_pos_idx, movie_label] = movie_maker

	def make_postfluor_movie(self, movie_label, fluor_channel,
		col_shading_alpha,
		bound_width,
		normalize_intensity, expansion_pixels = 10, bitdepth = None):
		'''
		Sets up movies for main (brightfield or phase contrast) channel 
		followed by postphase fluorescence whose channel_label is
		fluor_channel

		movie_label must be a string that isn't 'blank'		
		'''
		self._check_movie_label(movie_label)
		# initialize column in movie_obj_df for images for this movie
		self.movie_obj_df[movie_label] = None
		for xy_pos_idx in self.xy_positions:
			movie_maker = _PositionMovieMaker(
				self.analysis_config_obj_df,
				self.xy_pos_col_prop_dict[xy_pos_idx],
				col_shading_alpha,
				bound_width,
				normalize_intensity,
				expansion_pixels,
				postphase_fluor_channel = fluor_channel,
				bitdepth = bitdepth)
			# Keep movie_maker object in current df position
			# movies will be generated in combine_movie
			self.movie_obj_df.at[xy_pos_idx, movie_label] = movie_maker

	def make_fluor_movie(self, movie_label, fluor_channel, bound_width,
		normalize_intensity, expansion_pixels = 10, bitdepth = None):
		'''
		Sets up movies for fluorescent channel whose channel_label is
		fluor_channel

		movie_label must be a string that isn't 'blank'		
		'''
		self._check_movie_label(movie_label)
		# initialize column in movie_obj_df for images for this movie
		self.movie_obj_df[movie_label] = None
		# set col_shading_alpha to 0 (don't shade over fluorescent images)
		col_shading_alpha = 0
		for xy_pos_idx in self.xy_positions:
			movie_maker = _PositionMovieMaker(
				self.analysis_config_obj_df,
				self.xy_pos_col_prop_dict[xy_pos_idx],
				col_shading_alpha,
				bound_width,
				normalize_intensity,
				expansion_pixels,
				base_fluor_channel = fluor_channel,
				bitdepth = bitdepth)
			# Keep movie_maker object in current df position
			# movies will be generated in combine_movie
			self.movie_obj_df.at[xy_pos_idx, movie_label] = movie_maker

	def make_growth_plot_movie(self, movie_label, im_width, im_height,
		facet_override = None, add_growth_line = True, add_lag_line = True):
		'''
		Sets up movies for plot of growth over time

		movie_label must be a string that isn't 'blank'

		im_width is the (approx) width of the plot in pixels

		im_height is the (approx) height of the plot in pixels	
		'''
		self._check_movie_label(movie_label)
		# initialize column in movie_obj_df for images for this movie
		self.movie_obj_df[movie_label] = None
		for xy_pos_idx in self.xy_positions:
			movie_maker = _GrowthPlotMovieMaker(
				self.xy_pos_col_prop_dict[xy_pos_idx],
				self.analysis_config_obj_df,
				im_width,
				im_height,
				self.color_df,
				facet_override = facet_override,
				add_growth_line = add_growth_line,
				add_lag_line = add_lag_line)
			# Keep movie_maker object in current df position
			# movies will be generated in combine_movie
			self.movie_obj_df.at[xy_pos_idx, movie_label] = movie_maker

	def make_property_plot_movie(self, movie_label, im_width, im_height,
		plot_prop, facet_override = None, y_label_override = None):
		'''
		Sets up movies for plot of plot_prop over time

		movie_label must be a string that isn't 'blank'

		plot_prop can be the name of any column in
		colony_properties_combined file

		im_width is the (approx) width of the plot in pixels

		im_height is the (approx) height of the plot in pixels	
		'''
		self._check_movie_label(movie_label)
		# initialize column in movie_obj_df for images for this movie
		self.movie_obj_df[movie_label] = None
		for xy_pos_idx in self.xy_positions:
			movie_maker = _PlotMovieMaker(
				plot_prop,
				self.xy_pos_col_prop_dict[xy_pos_idx],
				self.analysis_config_obj_df,
				im_width,
				im_height,
				facet_override = facet_override,
				y_label_override = y_label_override)
			# Keep movie_maker object in current df position
			# movies will be generated in combine_movie
			self.movie_obj_df.at[xy_pos_idx, movie_label] = movie_maker

	def combine_movie(self, movie_width, movie_height,
		ordered_movie_components, component_size_dict, movie_format,
		blank_color = '#FFFFFF', duration = 1000, loop = 1,
		jpeg_quality = 95):
		'''
		Combine movies

		movie_width is the width (in pixels) of the final movie images

		movie_height is the height (in pixels) of the final movie images

		ordered_component_string is a string of movie labels with commas
		separating components in the same row of the movie frame and
		semicolons separating rows, and blank representing empty spaces
		(e.g. 'bf_movie | blank; rfp_movie|gfp_movie; growth_plot' has 3
		rows, with bf_movie in the upper left and nothing in the upper
		right, rfp_movie and gfp_movie in the second row, and growth_plot
		taking up the entire bottom row

		component_size_dict is a dict with keys as labeled movie components
		and values being (w, h) tuples for the size of each component
		row and column values must add up to im_height and im_width,
		respectively, although blank spacers will be used to take up any
		blank space (multiple blank values per row, or multiple instances of
		blank values taking up entire rows, will be sized equally in the x-
		or y-direction, respectively)

		blank_color is the hex value for the color of empty spaces between
		movies (default is white, #FFFFFF)

		movie_format can be 'jpeg', 'tiff', or 'gif'

		If movie_format is jpeg, can provide option jpeg_quality
		(default = 100)

		movie_format can be a sting or a list (for multiple output
		formats)
		If movie_format is gif, can provide options duration (time in
		milliseconds; default is 1000) and loop (number of loops; 0
		means forever, None means no looping)
		'''
		subframe_combiner = _SubframeCombiner(movie_width, movie_height,
			ordered_movie_components, component_size_dict, blank_color)
		# create movie for every xy position
		for xy_pos_idx in self.xy_positions:
			current_subframe_dict = dict()
			for movie_label in self.movie_obj_df.columns:
				movie_maker = \
					self.movie_obj_df.at[xy_pos_idx, movie_label]
				current_subframe_dict[movie_label] = \
					movie_maker.generate_movie_ims()
			compiled_movie_dict = \
				subframe_combiner.combine_subframes(current_subframe_dict)
			if isinstance(movie_format, list):
				for format in movie_format:
					self._save_movie(compiled_movie_dict, xy_pos_idx, format,
						duration, loop, jpeg_quality)
			else:
				self._save_movie(compiled_movie_dict, xy_pos_idx, movie_format,
					duration, loop, jpeg_quality)

def _generate_colony_colors(unique_colony_id_list,
	unique_colony_hex_color_list, randomize_order = True):
	'''
	Combines unique_colony_hex_color_list into a df with
	unique_colony_id_list, or, if unique_colony_hex_color_list is
	None, generates a list of hex color keys using p9

	Returns df with hex and RGB color values for each colony
	'''
	# from p9: https://p9.readthedocs.io/en/stable/_modules/p9/scales/scale_color.html
	# mimics p9 colors
	color_generator = hue_pal(h=.01, l=.6, s=.65, color_space='hls')
	color_num = len(unique_colony_id_list)
	if unique_colony_hex_color_list is None:
		if color_num == 1:
			hex_col = ['#fab514']
		else:
			hex_col = color_generator(color_num)
	else:
		hex_col = unique_colony_hex_color_list
	if randomize_order:
		random.shuffle(hex_col)
	color_df = pd.DataFrame({
		'cross_phase_tracking_id': unique_colony_id_list,
		'hex_color': hex_col,
		'rgb_color': [ImageColor.getcolor(c, "RGB") for c in hex_col]
		})
	return(color_df)
