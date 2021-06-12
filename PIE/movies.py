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
from itertools import chain
from PIL import ImageColor, Image
from PIE.image_coloring import \
	paint_by_numbers, \
	colorize_im, \
	overlay_color_im, \
	get_boundary, \
	safe_uint8_convert
from PIE.analysis_configuration import check_passed_config, process_setup_file
from PIE.colony_prop_compilation import get_colony_properties
from PIE.ported_matlab import bwperim

### !!! NEEDS UNITTESTS FOR THE WHOLE THING

### TODO: 	add text
####		change channel to use channel name from df, not file (since file channel names can switch between phases)
####		add scalebar
####		way to export plot data?
####		fix overlay to separate colony bounds and image

class MovieHolder(object):

	def __init__(self, global_tp_list):
		'''
		Generate an empty dataframe for holding output images
		'''
		self.im_df = pd.DataFrame(
			columns = ['global_timepoint', 'im'])
		self.im_df.global_timepoint = \
			np.sort(np.unique(np.array(global_tp_list)))
		self.im_df.set_index('global_timepoint', inplace = True)

	def add_im(self, global_timepoint, im):
		self.im_df.at[global_timepoint, 'im'] = im

class _MovieSaver(object):
	'''
	Parent object for classes that need to save movies
	'''

	def _get_ordered_image_list(self, compiled_movie_df):
		'''
		Puts images from compiled_movie_df into list in correct order
		'''
		compiled_movie_df_sorted = compiled_movie_df.sort_index()
		images_to_save = compiled_movie_df_sorted.im.to_list()
		return(images_to_save)

	def _write_video(self, compiled_movie_df, output_path, duration, fourcc):
		'''
		Writes movie of fourcc format using cv2
		'''
		# make sure images saved in order
		images_to_save = self._get_ordered_image_list(compiled_movie_df)
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

	def _write_gif(self, compiled_movie_df, output_path, duration, loop):
		'''
		Writes gif to output_path from images in compiled_movie_df

		duration is duration of each frame in milliseconds

		loop is the number of times gif loops
		(0 means forever, 1 means no looping)
		'''
		# make sure images saved in order
		images_to_save = self._get_ordered_image_list(compiled_movie_df)
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

	def _write_jpeg(self, compiled_movie_df, output_dir, jpeg_quality):
		'''
		Saves movie as jpegs in output_dir
		'''
		for tp_idx, row in compiled_movie_df.iterrows():
			filename = os.path.join(output_dir, str(tp_idx)+'.jpg')
			row.im.save(filename, quality = jpeg_quality)

	def _write_tiff(self, compiled_movie_df, output_dir):
		'''
		Saves movie as tiffs in output_dir
		'''
		for tp_idx, row in compiled_movie_df.iterrows():
			filename = os.path.join(output_dir, str(tp_idx)+'.tif')
			row.im.save(filename)

	def _save_movie(self, compiled_movie_df, movie_output_path, movie_name, movie_format,
		duration, loop, jpeg_quality):
		'''
		Saves movie for each movie_name in format specificed by
		movie_format, which can be 'jpeg'/'jpg', 'tiff'/'tif', 'gif',
		or video codecs ('h264' or 'mjpg'/'mjpeg', which both save to 
		.mov format)
		'''
		movie_format = movie_format.lower()
		if movie_format in ['tiff', 'tif', 'jpeg', 'jpg']:
			output_dir = os.path.join(movie_output_path, str(movie_name))
			if not os.path.isdir(output_dir):
				os.makedirs(output_dir)
			if movie_format in ['tiff', 'tif']:
				self._write_tiff(compiled_movie_df, output_dir)
			elif movie_format in ['jpg', 'jpeg']:
				self._write_jpeg(
					compiled_movie_df, output_dir, jpeg_quality)
		elif movie_format == 'gif':
			output_path = \
				os.path.join(self.movie_output_path,
					str(movie_name) + '.gif')
			self._write_gif(
				compiled_movie_df, output_path, duration, loop)
		elif movie_format == 'h264':
			output_path = \
				os.path.join(self.movie_output_path,
					str(movie_name) + '.mov')
			fourcc = cv2.VideoWriter_fourcc(*'H264')
			self._write_video(
				compiled_movie_df, output_path, duration, fourcc)
		elif movie_format in ['mjpeg', 'mjpg']:
			output_path = \
				os.path.join(self.movie_output_path,
					str(movie_name) + '.mov')
			fourcc = cv2.VideoWriter_fourcc(*'MJPG')
			self._write_video(
				compiled_movie_df, output_path, duration, fourcc)
		else:
			raise ValueError('Unrecognized movie format ' + movie_format)

	def _calc_inherent_size(self):
		'''
		Calculates default size of movie images
		'''
		self.inherent_size = (None, None)

	def generate_movie_ims(self, width, height, blank_color):
		'''
		Generate a dictionary of movie images with global timepoints 
		as keys
		'''
		pass

	def write_movie(self,
		movie_format, movie_output_path, movie_name,
		movie_width, movie_height,
		blank_color = 'white', duration = 1000, loop = 0,
		jpeg_quality = 95):
		'''
		Combine movies

		movie_width is the width (in pixels) of the final movie images; 
		if None, defaults to inherent movie width

		movie_height is the height (in pixels) of the final movie images;
		if None, defaults to inherent movie height

		blank_color is the color of empty spaces between movies 
		(default is white)

		movie_format can be 'jpeg', 'tiff', 'gif', 'h264', or 'mjpeg'

		If movie_format is jpeg or mjpeg, can pass option jpeg_quality
		(default = 95)

		movie_format can be a string or a list (for multiple output
		formats)

		If movie_format is gif, 'h264', or 'mjpeg', can pass optional 
		duration (time in milliseconds; default is 1000)

		For gif, can also pass loop (number of loops; 0 means forever, 
		None means no looping)

		movie_output_path is the folder in which the movie will be saved

		For video/gif outputs, file will be saved in 
		movie_output_path/movie_name.ext
		(e.g. movie_output_path/movie_name.gif); for tif/jpeg outputs,
		individual images will be saved inside
		movie_output_path/movie_name/
		'''
		# create movie output path
		if not os.path.isdir(movie_output_path):
			os.makedirs(movie_output_path)
		self.movie_output_path = movie_output_path
		if movie_height is None:
			if self.inherent_size[1] is None:
				raise ValueError('Must pass movie_height parameter')
			else:
				movie_height = self.inherent_size[1]
		if movie_width is None:
			if self.inherent_size[1] is None:
				raise ValueError('Must pass movie_width parameter')
			else:
				movie_width = self.inherent_size[0]
		compiled_movie_df = \
			self.generate_movie_ims(movie_width, movie_height, blank_color)
		if isinstance(movie_format, list):
			for format in movie_format:
				self._save_movie(compiled_movie_df, movie_output_path, movie_name, format,
					duration, loop, jpeg_quality)
		else:
			self._save_movie(compiled_movie_df, movie_output_path, movie_name, movie_format,
				duration, loop, jpeg_quality)

class _MovieMaker(_MovieSaver):
	'''
	Parent class for objects that make movies
	'''
	def __init__(self, global_timepoints, analysis_config_obj_df):
		self.analysis_config_obj_df = analysis_config_obj_df
		# set up movie_holder
		self.global_timepoints = global_timepoints
		# set up inherent movie size
		self._calc_inherent_size()

class _PlotMovieMaker(_MovieMaker):
	'''
	Plots required timepoints based on data in col_prop_df

	If facet_override is a bool, it overrides self._facet_phase_default
	w.r.t. whether plot should be faceted by phase
	'''
	def __init__(self, plot_prop, col_prop_df,
		analysis_config_obj_df,
		facet_override = None, y_label_override = None):
		self.plot_prop = plot_prop
		super(_PlotMovieMaker, self).__init__(
			col_prop_df.global_timepoint,
			analysis_config_obj_df
			)
		# add first timepoint time to col_prop_df
		self.col_prop_df = self._add_first_timepoint_data(col_prop_df)
		# add time in hours to self.col_prop_df
		self.col_prop_df['time_in_hours'] = \
			(self.col_prop_df['time_in_seconds'] - \
				self.first_phase_first_tp).astype(float)/3600
#				self.col_prop_df['first_timepoint_time']).astype(float)/3600
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
		self.first_phase_first_tp = self.first_tp_df.at[
			self.first_tp_df.index.min(),
			'first_timepoint_time'
			]
		self.first_tp_df['rel_first_timepoint'] = \
			self.first_tp_df.first_timepoint_time - self.first_phase_first_tp
		self.first_tp_df['rel_first_timepoint_hrs'] = \
			self.first_tp_df.rel_first_timepoint/3600
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
		# set up dataframe for plotting
		# convert phase_num to string for plotting
		df_to_plot = self.col_prop_df.copy()
		df_to_plot.phase_num = df_to_plot.phase_num.astype(str)
		# set up transparency column based on timepoint
		# make current value solid, previous values opaque, future 
		# values invisible
		# only make points transparent until the final timepoint
		if int(last_glob_tp) == \
			df_to_plot.global_timepoint.astype(int).max():
			df_to_plot['pt_time'] = 'present'
		else:
			df_to_plot['pt_time'] = 'future'
			df_to_plot.loc[
				df_to_plot.global_timepoint < last_glob_tp, 'pt_time'
				] = 'past'
			df_to_plot.loc[
				df_to_plot.global_timepoint == last_glob_tp, 'pt_time'
				] = 'present'
		# pass full data to ggplot for scales
		ggplot_obj = \
			p9.ggplot(data = df_to_plot) + \
			p9.geom_point(
				mapping = p9.aes(
					x = 'time_in_hours',
					y = self.plot_prop,
					color = 'hex_color',
					shape = 'phase_num',
					alpha = 'pt_time'),
				stroke = 0,
				size = 3,
				) + \
			p9.scale_colour_identity() + \
			p9.scale_alpha_manual(
				values={'future':0,'past':0.25, 'present':1}
				) + \
			p9.guides(
				alpha = False,
				shape = p9.guide_legend(title='phase')
				) + \
			p9.theme_bw() + \
			p9.theme(
				plot_margin = 0
				)
		# determine whether to facet by on phase_num
		if self._facet_phase:
			ggplot_obj = ggplot_obj + \
				p9.facet_wrap(
					'~phase_num', scales = 'free_x'
					)
		# if faceting by phase, or number of phases is 1, remove shape
		# legend
		if self._facet_phase or len(df_to_plot.phase_num.unique()) == 1:
			ggplot_obj = ggplot_obj + \
				p9.guides(shape = False)
		# add x and y axis properties
		min_time = self.col_prop_df['time_in_hours'].min()
		max_time = self.col_prop_df['time_in_hours'].max()
		time_span = max_time-min_time
		ggplot_obj = ggplot_obj + \
			p9.scale_x_continuous(
				name = 'time (hours)',
				limits = (
					min_time-time_span/20,
					max_time+time_span/20
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

	def _plot_to_im(self, ggplot_obj, width, height):
		'''
		Returns Image object of ggplot_obj plot

		Warning: height and width are not exactly provided height and
		width (consistent with ggplot issues)
		'''
		# modified from https://stackoverflow.com/a/58641662/8082611
		buf = BytesIO()
		# create fake width and height based on pixels
		# this parameter actually ends up affecting text size on the 
		# plot...... and so needs to be kept relatively consistent
		fake_dpi = 300
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			ggplot_obj.save(
				filename=buf,
				format='png',
				width=width/fake_dpi,
				height=height/fake_dpi,
				units='in', dpi=fake_dpi,
				limitsize=False)
		buf.seek(0)
		img_cv2 = cv2.imdecode(
			np.frombuffer(buf.getvalue(), dtype=np.uint8), 1
			)
		img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB), 'RGB')
#		img = Image.frombuffer('RGB', (width, height), buf.getvalue(), 'raw', 'RGB', 0, 1)
		buf.close()
#		img = cv2.imdecode(img_arr, 1)
#		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img

	def _create_plot(self, last_glob_tp, width, height):
		'''
		Creates plot as image, plotting all points up to and including
		last_glob_tp
		'''
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			ggplot_obj = self._plot_property(last_glob_tp)
		plot_im = self._plot_to_im(ggplot_obj, width, height)
		return(plot_im)

	def generate_movie_ims(self, width, height, blank_color):
		self.movie_holder = MovieHolder(self.global_timepoints)
		for global_tp in self.col_prop_df.global_timepoint.unique():
			plot_im = self._create_plot(global_tp, width, height)
			self.movie_holder.add_im(global_tp, plot_im)
		return(self.movie_holder.im_df)

class _GrowthPlotMovieMaker(_PlotMovieMaker):
	'''
	Plots growth across required timepoints based on data in col_prop_df

	If facet_override is a bool, it overrides self._facet_phase_default
	w.r.t. whether plot should be faceted by phase
	'''

	def __init__(self, col_prop_df, color_df,
		analysis_config_obj_df,
		facet_override = None, add_growth_line = True, add_lag_line = True):
		self.add_growth_line = add_growth_line
		self.add_lag_line = add_lag_line
		self.color_df = color_df
		super(_GrowthPlotMovieMaker, self).__init__(
			'ln_area', col_prop_df, analysis_config_obj_df,
			facet_override, y_label_override = 'ln(area)')

	def _set_start_end_times_and_log_areas(self, gr_df):
		'''
		Set times for starting and ending growth tracking in gr_df
		'''
		# use timings from self.col_prop_df to set time_in_hrs  of t0
		# and tfinal
		key_df_time = \
			self.timing_df[['timepoint', 'time_in_hours', 'phase_num']]
		t0_key = key_df_time.rename(
			columns={
				'timepoint': 't0',
				'time_in_hours': 't0_in_hours'
				})
		tfinal_key = key_df_time.rename(
			columns={
				'timepoint': 'tfinal',
				'time_in_hours': 'tfinal_in_hours'
				})
		gr_df = pd.merge(left = gr_df, right = t0_key)
		gr_df = pd.merge(left = gr_df, right = tfinal_key)
		# add first timepoint info
		gr_df = pd.merge(
			left = gr_df,
			right = self.first_tp_df[['phase_num','rel_first_timepoint_hrs']]
			)
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
		gr_df['yhat_t0'] = \
			(gr_df.t0_in_hours - gr_df.rel_first_timepoint_hrs) * \
			gr_df.gr + gr_df.intercept
		gr_df['yhat_tfinal'] = \
			(gr_df.tfinal_in_hours - gr_df.rel_first_timepoint_hrs) * \
			gr_df.gr + gr_df.intercept
		gr_df['lag_end'] = gr_df.lag + gr_df.rel_first_timepoint_hrs
		return(gr_df)

	def _add_first_tp_colony_size(self, gr_df):
		'''
		Add the log area of the colony at the first timepoint
		'''
		# create dataframe of first timepoint for every colony in 
		# every phase
		first_timepoint_key_df = self.col_prop_df.groupby(
			['phase_num','cross_phase_tracking_id']
			)['timepoint'].min()
		# subset col_prop_df where timepoint equals first timepoint
		col_prop_df_first_tp = \
			pd.merge(left = self.col_prop_df, right = first_timepoint_key_df)
		col_prop_df_first_tp.rename(
			columns={'ln_area': 'ln_area_first_tp'},
			inplace = True
			)
		gr_df = pd.merge(
			left = gr_df,
			right = col_prop_df_first_tp[
				['phase_num','cross_phase_tracking_id','ln_area_first_tp']
				]
			)
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
		# add log(area) at first timepoint to gr_df
		gr_df_first_tp_area = self._add_first_tp_colony_size(gr_df_extra_times)
		# change phase_num to string type
		gr_df_first_tp_area.phase_num = gr_df_first_tp_area.phase_num.astype(str)
		self.gr_df = gr_df_first_tp_area

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
							x = 'rel_first_timepoint_hrs',
							y = 'ln_area_first_tp',
							xend = 'lag_end',
							yend = 'ln_area_first_tp',
							color = 'hex_color'),
						size = 1,
						linetype = 'dashed')
		return(ggplot_obj)

	def _create_plot(self, last_glob_tp, width, height):
		'''
		Creates plot as image, plotting all points up to and including
		last_glob_tp
		'''
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			ggplot_obj = self._plot_growth_rate(last_glob_tp)
		plot_im = self._plot_to_im(ggplot_obj, width, height)
		return(plot_im)

class _ImMovieMaker(_MovieMaker):
	'''
	Parent class for _MovieMaker classes that generate movies from
	images (rather than plots)
	'''
	def __init__(self, global_timepoints, analysis_config_obj_df):
		super(_ImMovieMaker, self).__init__(
			global_timepoints,
			analysis_config_obj_df
			)
		# check data
		self._perform_data_checks()

	def _create_blank_subframe(self, third_dim = False):
		'''
		Creates a black cv2 8-bit image of self.inherent_size

		If third_dim is True, adds third dimension
		'''
		blank_subframe = np.uint8(np.zeros(
			shape = (
				self.inherent_size[1],
				self.inherent_size[0]
				)
			))
		if third_dim:
			blank_subframe = np.dstack([blank_subframe]*3)
		return(blank_subframe)

	def _perform_data_checks():
		pass

	def generate_postphase_frame(self, phase):
		'''
		Generates a boundary image for postphase fluorescent image
		'''
		pass

	def generate_frame(self, global_timepoint):
		'''
		Generates a boundary image for global_timepoint
		'''
		pass

class _PositionMovieMaker(_ImMovieMaker):
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

	main_phase_color and postphase_color are the colors in which the 
	maximum-intensity pixels will be displayed for the main channel
	and the postphase channel; all other pixels will be displayed in
	a gradient from black to that color. Can be specified as a hex 
	code, or as a color name for many colors (e.g. 'white', 'magenta', 
	etc)

	base_fluor_channel is the fluorescent channel name to use for images;
	if None (default) uses main_channel_label

	normalize_intensity is a bool that determines whether input image
	should be normalized; if True, the image bitdepth must be passed

	bitdepth is an optional argument for the bitdepth of the input image
	'''
	def __init__(self, col_prop_df,
		col_shading_alpha, bound_width, normalize_intensity,
		expansion_pixels, 
		analysis_config_obj_df,
		main_phase_color = 'white',
		postphase_color = 'white',
		base_fluor_channel = None, bitdepth = None,
		postphase_fluor_channel = None):
		self.col_prop_df = col_prop_df
		self.xy_pos_idx = \
			self._get_single_val_from_df('xy_pos_idx', col_prop_df)
		self.normalize_intensity = bool(normalize_intensity)
		self.base_fluor_channel = base_fluor_channel
		self.postphase_fluor_channel = postphase_fluor_channel
		self.expansion_pixels = int(expansion_pixels)
		self.col_shading_alpha = col_shading_alpha
		self.bound_width = bound_width
		self.bitdepth = bitdepth
		self.main_phase_color = ImageColor.getcolor(main_phase_color, "RGB")
		self.postphase_color = ImageColor.getcolor(postphase_color, "RGB")
		# identify bounds of movie in image
		# (assume all images are the same size)
		self._id_im_bounds(analysis_config_obj_df)
		# calculate inherent size and set up MovieHolder
		_ImMovieMaker.__init__(
			self,
			self.col_prop_df.global_timepoint,
			analysis_config_obj_df
			)

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

	def _id_im_bounds(self, analysis_config_obj_df):
		'''
		Identifies bounding pixels of movie
		'''
		temp_analysis_config = analysis_config_obj_df.iloc[0]['analysis_config']
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

	def _calc_inherent_size(self):
		'''
		Calculates default size of movie images
		'''
		im_width = self.x_range_end - self.x_start
		im_height = self.y_range_end - self.y_start
		self.inherent_size = (im_width, im_height)

	def _create_subframe(self, frame):
		'''
		Crops frame based on values created in self._id_im_bounds
		'''
		subframe = \
			frame[self.y_start:self.y_range_end, self.x_start:self.x_range_end]
		return(subframe)

	def _subframe_and_colorize(self, im, im_color):
		'''
		Crops im based on bounds and returns normalized, colorized 
		8-bit np array

		color of image is black for 0 intensity and im_color for max 
		possible intensity
		'''
		if im is None:
			im_8_bit = self._create_blank_subframe()
		else:
			# subframe mask and input im
			input_im = self._create_subframe(im)
			# convert input_im to 8-bit, normalizing if necessary
			if self.normalize_intensity:
				im_8_bit = cv2.normalize(input_im, None, alpha=0, beta=(2**8-1),
					norm_type=cv2.NORM_MINMAX)
			else:
				# change bitdepth to 8-bit
				im_8_bit = \
					np.uint8(np.round(input_im.astype(float)/(2**(self.bitdepth - 8))))
		# initialize image to overlay and colorize if necessary
		im_colorized = colorize_im(im_8_bit, im_color)
		return(im_colorized)

	def _get_fluor_property(self, channel_name, analysis_config, fluor_property):
		'''
		Gets the fluor_property corresponding to channel_name in 
		analysis_config
		'''
		fluor_df = analysis_config.fluor_channel_df.copy()
		if channel_name in fluor_df.fluor_channel_column_name.to_list():
			fluor_df.set_index('fluor_channel_column_name', inplace = True)
			prop = fluor_df.at[channel_name, fluor_property]
		else:
			prop = None
		return(prop)

	def _get_glob_tp_data(self, global_timepoint):
		'''
		Returns timepoint and analysis_config associated with 
		global_timepoint
		'''
		tp_col_prop_df = self.col_prop_df[
			self.col_prop_df.global_timepoint == global_timepoint
			]
		phase = self._get_single_val_from_df('phase_num', tp_col_prop_df)
		timepoint = int(
			self._get_single_val_from_df('timepoint', tp_col_prop_df)
			)
		analysis_config = \
			self.analysis_config_obj_df.at[phase, 'analysis_config']
		analysis_config.set_xy_position(self.xy_pos_idx)
		return(timepoint, analysis_config)

	def _make_label_key_dict(self, col_prop_df):
		'''
		Makes a dict with keys from col_prop_df.label and values 
		(r,g,b) tuples from self.col_prop_df.rgb_color
		'''
		# get col_prop_df for current timepoint
		label_color_dict = {}
		for col_row in col_prop_df.itertuples():
			# create list of labels
			current_label = col_row.label
			if isinstance(current_label, str):
				label_list = [int(l) for l in current_label.split(';')]
			else:
				label_list = [int(current_label)]
			# get rgb tuple
			rgb_val = col_row.rgb_color
			rgb_val_list = [rgb_val]*len(label_list)
			# make dict of labels and copies of current rgb tuple
			current_dict = dict(zip(label_list, rgb_val_list))
			# add dict to full dictionary
			label_color_dict.update(current_dict)
		return(label_color_dict)

	def _generate_color_overlays_confirmed(self, timepoint, analysis_config, 
		tp_col_prop_df):
		'''
		Returns colony and boundary overlay for timepoint from 
		analysis_config, for colonies in tp_col_prop_df
		'''
		# read in mask
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
		labeled_mask_full = \
			cv2.imread(colony_mask_path, cv2.IMREAD_ANYDEPTH)
		# crop mask
		labeled_mask = self._create_subframe(labeled_mask_full)
		# create labeled mask of boundaries
		bool_mask = labeled_mask > 0
		boundary_mask = get_boundary(bool_mask, self.bound_width)
		labeled_boundary_mask = labeled_mask*np.uint8(boundary_mask)
		# get dictionary of labels and colors
		label_key_dict = self._make_label_key_dict(tp_col_prop_df)
		# get colored colony and boundary masks
		colored_colony_mask = \
			paint_by_numbers(labeled_mask, label_key_dict)
		colored_boundary_mask = \
			paint_by_numbers(labeled_boundary_mask, label_key_dict)
		return(colored_colony_mask, colored_boundary_mask)

	def generate_color_overlays(self, global_timepoint):
		'''
		Returns colony and boundary overlay for global_timepoint
		'''
		# get labeled colony mask and input file
		if global_timepoint in self.col_prop_df.global_timepoint.values:
			# get timepoint and analysis_config
			timepoint, analysis_config = \
				self._get_glob_tp_data(global_timepoint)
			# get col_prop_df for current timepoint
			tp_col_prop_df = self.col_prop_df[
				self.col_prop_df.global_timepoint == global_timepoint
				]
			colored_colony_mask, colored_boundary_mask = \
				self._generate_color_overlays_confirmed(
					timepoint, analysis_config, tp_col_prop_df
					)
		else:
			colored_colony_mask = self._create_blank_subframe(third_dim = True)
			colored_boundary_mask = colored_colony_mask.copy()
		return(colored_colony_mask, colored_boundary_mask)

	def generate_raw_frame(self, global_timepoint):
		'''
		Generates image subframe for global_timepoint without colony 
		recognition image overlay, and colorizes it based on 
		self.main_phase_color

		returns np.uint8 array
		'''
		if global_timepoint in self.col_prop_df.global_timepoint.values:
			# get timepoint and analysis_config
			timepoint, analysis_config = \
				self._get_glob_tp_data(global_timepoint)
			# get channel_label to use
			if self.base_fluor_channel is None:
				channel_label = analysis_config.main_channel_label
			else:
				channel_label = self._get_fluor_property(
					self.base_fluor_channel,
					analysis_config,
					'fluor_channel_label'
					)
			# get input image
			if channel_label is None:
				input_im_full = None
			else:
				input_im_full, _, _ = \
					analysis_config.get_image(timepoint, channel_label)
			# get cropped, normalized, colorized image
			im_colorized = self._subframe_and_colorize(
				input_im_full, self.main_phase_color
				)
		else:
			im_colorized = self._create_blank_subframe(third_dim = True)
		return(im_colorized)

	def generate_frame(self, global_timepoint):
		'''
		Generates a boundary image for global_timepoint
		'''	
		# initialize image subframe and colorize if necessary
		im = self.generate_raw_frame(global_timepoint)
		colored_colony_mask, colored_boundary_mask = \
			self.generate_color_overlays(global_timepoint)
		# shade colonies
		if self.col_shading_alpha == 0:
			shaded_im = im
		else:
			shaded_im = overlay_color_im(
				im,
				colored_colony_mask,
				self.col_shading_alpha
				)
		# add colony boundary
		im_with_overlay = overlay_color_im(
			shaded_im,
			colored_boundary_mask,
			1
			)
		im_8bit = safe_uint8_convert(im_with_overlay)
		# convert im_with_overlay to Pillow Image object
		im_with_overlay_pil = \
			Image.fromarray(im_8bit, 'RGB')
		return(im_with_overlay_pil)

	def generate_postphase_raw_frame(self, phase):
		'''
		Generates image subframe for postphase fluorescence for phase, 
		and colorizes it based on self.main_phase_color

		returns np.uint8 array
		'''
		postphase_analysis_config = \
			self.analysis_config_obj_df.at[phase,
				'postphase_analysis_config']
		if postphase_analysis_config is None or \
			self.postphase_fluor_channel is None:
			global_tp = None
			im_colorized = None
		else:
			postphase_analysis_config.set_xy_position(self.xy_pos_idx)
			# get input image for postphase fluorescent channel
			postphase_fluor_channel_label = \
				self._get_fluor_property(
					self.postphase_fluor_channel,
					postphase_analysis_config,
					'fluor_channel_label'
					)
			if postphase_fluor_channel_label is None:
				input_im_full = None
			else:
				input_im_full, _, _ = \
					postphase_analysis_config.get_image(
						None,
						postphase_fluor_channel_label
						)
			# get cropped, normalized, colorized image
			im_colorized = self._subframe_and_colorize(
				input_im_full, self.postphase_color)
		return(im_colorized)

	def generate_postphase_color_bounds(self, phase):
		'''
		Returns boundary overlay for postphase
		'''
		postphase_analysis_config = \
			self.analysis_config_obj_df.at[phase,
				'postphase_analysis_config']
		# get postphase image if necessary
		if postphase_analysis_config is None or \
			self.postphase_fluor_channel is None:
			colored_boundary_mask = \
				self._create_blank_subframe(third_dim = True)
		else:
			analysis_config = \
				self.analysis_config_obj_df.at[phase,
					'analysis_config']
#			postphase_analysis_config.set_xy_position(self.xy_pos_idx)
			analysis_config.set_xy_position(self.xy_pos_idx)
			# get timepoint at which to draw boundary for each colony
			timepoint_label = \
				self._get_fluor_property(
					self.postphase_fluor_channel,
					postphase_analysis_config,
					'fluor_timepoint'
					)			
			# make df that holds label, timepoint to use, and
			# cross_phase_tracking_id for each colony in current phase
			phase_col_prop_df = self.col_prop_df[
				self.col_prop_df.phase_num == phase
				]
			colony_df = phase_col_prop_df[
				['time_tracking_id', 'cross_phase_tracking_id', 'rgb_color']
				].drop_duplicates()
			# get timepoint at which to draw boundary for each colony
			timepoint_label = \
				self._get_fluor_property(
					self.postphase_fluor_channel,
					postphase_analysis_config,
					'fluor_timepoint'
					)
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
			# get timepoint labels from current phase
			# need to import label colony property mat dataframe
			label_property_mat_path = \
				analysis_config.get_property_mat_path('label')
			label_property_mat_df = pd.read_csv(label_property_mat_path, index_col = 0)
			colony_df['label'], colony_df['timepoint'] = \
				get_colony_properties(
					label_property_mat_df,
					timepoint_label,
					index_names = colony_df.time_tracking_id.to_list(),
					gr_df = gr_df
					)
			colored_boundary_mask = self._create_blank_subframe(third_dim = True)
			for timepoint in colony_df.timepoint.unique():
				# get part of colony_df corresponding to the current
				# timepoint
				current_col_prop_df = \
					colony_df[
						colony_df.timepoint == timepoint
						]
				_, current_boundary_mask = \
					self._generate_color_overlays_confirmed(
						timepoint,
						analysis_config,
						current_col_prop_df
						)
				# add current colony boundary mask to full boundary mask
				colored_boundary_mask = overlay_color_im(
					colored_boundary_mask, current_boundary_mask, 1
					)
		return(colored_boundary_mask)

	def get_postphase_global_tp(self, phase):
		'''
		Returns global_timepoint for postphase of current phase
		'''
		postphase_analysis_config = \
			self.analysis_config_obj_df.at[phase,
				'postphase_analysis_config']
		if postphase_analysis_config is None or \
			self.postphase_fluor_channel is None:
			global_tp = None
		else:
			phase_col_prop_df = self.col_prop_df[
				self.col_prop_df.phase_num == phase
				]
			global_tp = phase_col_prop_df.global_timepoint.max() + 0.1
		return(global_tp)

	def generate_postphase_frame(self, phase):
		'''
		Generates a postphase fluorescent image with boundaries
		'''
		# get modified global_timepoint
		global_tp = self.get_postphase_global_tp(phase)
		if global_tp is None:
			im_with_overlay_pil = None
		else:
			# get raw fluor image
			colorized_im = self.generate_postphase_raw_frame(phase)
			colored_boundary_mask = self.generate_postphase_color_bounds(phase)
			# add colony boundary
			im_with_overlay = overlay_color_im(
				colorized_im,
				colored_boundary_mask,
				1
				)
			im_8bit = safe_uint8_convert(im_with_overlay)
			# convert im_with_overlay to Pillow Image object
			im_with_overlay_pil = \
				Image.fromarray(im_8bit, 'RGB')
		return(global_tp, im_with_overlay_pil)

	def generate_movie_ims(self, width, height, blank_color):
		self.movie_holder = MovieHolder(self.global_timepoints)
		# first add the regular images, then the postphase images
		for global_tp in self.col_prop_df.global_timepoint.unique():
			im_with_overlay = self.generate_frame(global_tp)
			# add image with overlay to movie_holder
			im_resized = \
				_ratio_resize_convert(
					im_with_overlay, width, height, blank_color
					)
			self.movie_holder.add_im(global_tp, im_resized)
		# add postphase analysis images if necessary
		for phase in self.analysis_config_obj_df.index:			
			global_tp, im_with_overlay = \
				self.generate_postphase_frame(phase)
			if im_with_overlay != None:
				# add image with overlay to movie_holder
				im_resized = \
					_ratio_resize_convert(
						im_with_overlay, width, height, blank_color
						)
				self.movie_holder.add_im(global_tp, im_resized)
		return(self.movie_holder.im_df)

class _OverlayMovieMaker(_ImMovieMaker):
	'''
	Overlays movies in self.overlay_df.movie_obj using relative 
	intensities in rel_gain_list

	self.overlay_df.movie_obj must all be _PositionMovieMaker class
	objects with identical inherent_size

	intens_mult_list is a list of multipliers for intensities with 
	which to display the pixels of every 'channel' in the blended 
	image; if None, sets all intensity multipliers to be 1

	If elements of self.overlay_df.movie_obj differ in their
	global_timepoints, missing global_timepoints will be treated as
	empty (all-black) images
	'''
	def __init__(self, movie_obj_list, intens_mult_list):
		if len(movie_obj_list) < 2:
			raise IndexError('movie_obj_list must contain 2 or more objects')
		intens_mult_list = self._check_intensity_multipliers(
			intens_mult_list, movie_obj_list
			)
		self.overlay_df = pd.DataFrame({
			'movie_obj':movie_obj_list,
			'intens_mult':intens_mult_list
			})
		global_timepoints = self._get_full_global_tp_list(movie_obj_list)
		analysis_config_obj_df = \
			self._get_analysis_config_obj_df(movie_obj_list)
		super(_OverlayMovieMaker, self).__init__(
			global_timepoints,
			analysis_config_obj_df,
			)
	def _check_intensity_multipliers(self, intens_mult_list, movie_obj_list):
		movie_obj_num = len(movie_obj_list)
		if intens_mult_list is None:
			intens_mult_list = \
				np.ones(shape = (1, movie_obj_num), dtype = float)
		else:
			if len(intens_mult_list) != movie_obj_num:
				raise IndexError(
					'Length of intensity multipliers and movie objects must '
					'have the same length'
					)
			intens_mult_list = np.array(intens_mult_list)
		return(intens_mult_list)

	def _get_full_global_tp_list(self, movie_obj_list):
		'''
		Gets full list of global timepoints across all objects in
		movie_obj_list
		'''
		global_timepoints = \
			list(set(chain(
				*[x.global_timepoints for x in movie_obj_list]
				)))
		return(global_timepoints)

	def _get_analysis_config_obj_df(self, movie_obj_list):
		'''
		Checks that analysis_config_obj_df is identical in all objects
		in movie_obj_list, and returns the df
		'''
		analysis_config_obj_df_list = \
			[x.analysis_config_obj_df for x in movie_obj_list]
		analysis_config_obj_df = analysis_config_obj_df_list[0]
		if not all([analysis_config_obj_df.equals(a)
			for a in analysis_config_obj_df_list[1:]]):
			raise ValueError('All objects in movie_obj_list must have '
				'identical analysis_config_obj dataframes')
		return(analysis_config_obj_df)

	def _perform_data_checks(self):
		movie_obj_list = self.overlay_df.movie_obj.to_list()
		intens_mult_list = self.overlay_df.intens_mult.to_numpy()
		if np.any(intens_mult_list < 0):
			raise ValueError('intensity multipliers must be >= 0')
		# check that movie_obj instances are of correct type
		if not all([isinstance(x,_PositionMovieMaker) for x in movie_obj_list]):
			raise TypeError(
				'All objects in movie_obj_list must be movie objects generated'
				' directly from images')

	def _calc_inherent_size(self):
		'''
		Calculates default size of movie images
		'''
		unique_inherent_sizes = list(set(
			[x.inherent_size for x in self.overlay_df.movie_obj.to_list()]
			))
		if len(unique_inherent_sizes) != 1:
			raise ValueError('inherent_size for all objects in movie_obj_list '
				'must be identical')
		else:
			self.inherent_size = unique_inherent_sizes[0]

	def generate_frame(self, global_timepoint):
		'''
		generates frame for global_tp
		'''
		# initialize blank images to accumulate onto
		comb_im = self._create_blank_subframe(third_dim = True).astype(float)
		comb_col_mask = comb_im.copy()
		comb_bound_mask = comb_im.copy()
		# create a multiplier for colony shading and boundaries
		equal_multiplier = 1.0/self.overlay_df.shape[0]
		mean_col_shading_alpha = 0
		for overlay_idx in self.overlay_df.index:
			mov_obj = self.overlay_df.at[overlay_idx, 'movie_obj']
			intens_mult = self.overlay_df.at[overlay_idx, 'intens_mult']
			im = mov_obj.generate_raw_frame(global_timepoint)
			curr_colony_mask, curr_boundary_mask = \
				mov_obj.generate_color_overlays(global_timepoint)
			# blend background frames in correct proportions
			comb_im = comb_im + im.astype(float)*intens_mult
			# blend colony and boundary masks in equal proportions
			comb_col_mask = \
				comb_col_mask + curr_colony_mask.astype(float)*equal_multiplier
			comb_bound_mask = \
				comb_bound_mask + curr_boundary_mask.astype(float)*equal_multiplier
			# accumulate mean colony shading alpha
			mean_col_shading_alpha = \
				mean_col_shading_alpha + \
				mov_obj.col_shading_alpha*equal_multiplier
		shaded_im = overlay_color_im(
			comb_im,
			comb_col_mask,
			mean_col_shading_alpha
			)
		# add colony boundary
		im_comb_with_overlay = overlay_color_im(
			shaded_im,
			comb_bound_mask,
			1
			)
		im_combined_8bit = safe_uint8_convert(im_comb_with_overlay)
		# convert to Pillow Image format first
		im_combined_pil = Image.fromarray(im_combined_8bit, 'RGB')
		return(im_combined_pil)

	def generate_postphase_frame(self, phase):
		'''
		Generates a boundary image for postphase fluorescent image
		'''
		# initialize blank images to accumulate onto
		comb_im = self._create_blank_subframe(third_dim = True).astype(float)
		comb_bound_mask = comb_im.copy()
		# create a multiplier for colony shading and boundaries
		equal_multiplier = 1.0/self.overlay_df.shape[0]
		# set up global_tp accumulator
		comb_global_tp = None
		for overlay_idx in self.overlay_df.index:
			mov_obj = self.overlay_df.at[overlay_idx, 'movie_obj']
			intens_mult = self.overlay_df.at[overlay_idx, 'intens_mult']
			# get modified global_timepoint
			global_tp = mov_obj.get_postphase_global_tp(phase)
			if global_tp is None:
				if comb_im is not None or comb_bound_mask is not None:
					if np.sum(comb_im) != 0 or np.sum(comb_bound_mask) != 0:
						warnings.warn(
							'Trying to merge movies with and without a '
							'postphase fluorescence experiment; removing '
							'postphase data',
							UserWarning
							)
				comb_im = None
				im = None
				comb_bound_mask = None
				curr_boundary_mask = None
			else:
				if comb_global_tp is None:
					comb_global_tp = global_tp
				else:
					if comb_global_tp != global_tp:
						raise ValueError(
							'surmised global postphase timepoint for '
							'postphase images not identical in merged movies'
							)
				# get raw fluor image
				im = mov_obj.generate_postphase_raw_frame(phase)
				curr_boundary_mask = mov_obj.generate_postphase_color_bounds(phase)
				# blend background frames in correct proportions
				comb_im = comb_im + im.astype(float)*intens_mult
				# blend boundary masks in equal proportions
				comb_bound_mask = \
					comb_bound_mask + curr_boundary_mask.astype(float)*equal_multiplier
		if comb_im is None:
			im_combined_pil = None
		else:
			# add colony boundary
			im_comb_with_overlay = overlay_color_im(
				comb_im,
				comb_bound_mask,
				1
				)
			im_combined_8bit = safe_uint8_convert(im_comb_with_overlay)
			# convert to Pillow Image format first
			im_combined_pil = Image.fromarray(im_combined_8bit, 'RGB')
		return(global_tp, im_combined_pil)

	def generate_movie_ims(self, width, height, blank_color):
		# first add the regular images, then the postphase images
		self.movie_holder = MovieHolder(self.global_timepoints)
		for global_tp in self.movie_holder.im_df.index:
			im_combined_pil = self.generate_frame(global_tp)
			im_resized = \
				_ratio_resize_convert(
					im_combined_pil, width, height, blank_color
					)
			self.movie_holder.add_im(global_tp, im_resized)
		# add postphase analysis images if necessary
		for phase in self.analysis_config_obj_df.index:
			global_tp, im_combined_pil_postphase = \
				self.generate_postphase_frame(phase)
			if im_combined_pil_postphase != None:
				# add image with overlay to movie_holder
				im_resized = \
					_ratio_resize_convert(
						im_combined_pil_postphase, width, height, blank_color
						)
				self.movie_holder.add_im(global_tp, im_resized)
		return(self.movie_holder.im_df)

class _MovieGrid(_MovieSaver):
	'''
	Creates a _MovieGrid object from a list of _MovieGrid- or 
	_MovieMaker-inheriting objects arranged in a single direction 
	(row or column)

	movie_obj_list can contain _MovieGrid or _MovieMaker objects

	grid_axis is the direction along which the grid is created; can be 
	either "row" or "column"

	grid_ratios contains a list of relative ratios of frame sizes along
	grid_axis; if None, relative ratios are identical for all 
	components of movie_obj_list
	'''
	def __init__(self, movie_obj_list, grid_axis, grid_ratios):
		# set up grid_ratios
		self.grid_ratios = _check_rel_ratios(grid_ratios, movie_obj_list)
		# set columns to modify along current grid_axis
		self._set_grid_prop_names(grid_axis)
		# create self.movie_maker_df and get data for inherent size calc
		self._create_grid(movie_obj_list)
		# calculate inherent size: the default is for the object with
		# the smallest image to be displayed at full resolution
		self._calc_inherent_size()

	def _set_grid_prop_names(self, grid_axis):
		self.grid_axis = grid_axis
		if self.grid_axis == 'row':
			self.grid_prop_name = 'width_prop'
			self.non_grid_prop_name = 'height_prop'
			self.grid_pos_name = 'left_pos'
			self.non_grid_pos_name = 'top_pos'
			self.grid_dim = 'width'
			self.non_grid_dim = 'height'
		elif self.grid_axis == 'column':
			self.non_grid_prop_name = 'width_prop'
			self.grid_prop_name = 'height_prop'
			self.non_grid_pos_name = 'left_pos'
			self.grid_pos_name = 'top_pos'
			self.non_grid_dim = 'width'
			self.grid_dim = 'height'
		else:
			raise ValueError('grid_axis must be "row" or "column"')

	def _create_grid(self, movie_obj_list):
		'''
		Creates grid going in one direction (either horizontally or 
		vertically)
		'''
		# add objects to grid df one by one
		grid_pos = 0
		width_list = []
		height_list = []
		movie_maker_df_list = []
		for movie_obj, grid_prop in zip(movie_obj_list, self.grid_ratios):
			if isinstance(movie_obj, _MovieMaker) or \
				isinstance(movie_obj, _MovieGrid):
				current_df = \
					self._get_df_movie_saver(movie_obj, grid_prop, grid_pos)
#			elif isinstance(movie_obj, _MovieGrid):
#				current_df = \
#					self._get_df_movie_grid(movie_obj, grid_prop, grid_pos)
			else:
				raise TypeError('movie objects must be _MovieMaker or '
					'_MovieGrid type objects')
			movie_maker_df_list.append(current_df)
			# update lists of movie object widths and heights
			width_list.append(movie_obj.inherent_size[0])
			height_list.append(movie_obj.inherent_size[1])
			# update starting position of image along grid
			grid_pos = grid_pos + grid_prop
		# create dataframe containing movie_maker objects, their grid 
		# proportions, and their starting positions on the grid
		self.movie_maker_df = \
			pd.concat(movie_maker_df_list, ignore_index = True)
		# set up df for inherent size calculation
		self._inherent_dim_df = \
			pd.DataFrame({'width': width_list, 'height': height_list})
		# replace None with NaN to aid further processing
		self._inherent_dim_df.fillna(value = np.nan, inplace = True)

	def _get_df_movie_saver(self, movie_maker_obj, grid_prop, grid_pos):
		'''
		Get dataframe containing _MovieMaker class object, the 
		proportion of the grid it takes up, and its start positions
		'''
		movie_maker_obj_df = pd.DataFrame({
			'movie_maker': movie_maker_obj,
			self.grid_prop_name: grid_prop,
			self.non_grid_prop_name: 1,
			self.grid_pos_name: grid_pos,
			self.non_grid_pos_name: 0
			},
			index = [0])
		return(movie_maker_obj_df)

#	def _get_df_movie_grid(self, movie_grid_obj, grid_prop, grid_pos):
#		'''
#		Get dataframe from _MovieGrid class object, and update the 
#		proportion of the full grid it takes up and its start positions
#		'''
#		movie_grid_obj_df = movie_grid_obj.movie_maker_df.copy()
#		movie_grid_obj_df[self.grid_prop_name] = \
#			movie_grid_obj_df[self.grid_prop_name]*grid_prop
#		movie_grid_obj_df[self.grid_pos_name] = \
#			movie_grid_obj_df[self.grid_pos_name]*grid_prop + grid_pos
#		return(movie_grid_obj_df)

	def _calc_inherent_size(self):
		'''
		Calculates default size of movie images
		'''
		inherent_size_ser = pd.Series(dtype=float)
		# along the grid dimension, each object in movie_grid_obj must 
		# fit within the ratio alotted to it
		inherent_size_ser[self.grid_dim] = \
			self._calc_max_single_dim_size(
				self._inherent_dim_df[self.grid_dim].to_numpy(),
				self.grid_ratios
				)
		# along the non-grid dimension, the inherent size is just the 
		# size of the largest object
		inherent_size_ser[self.non_grid_dim] = \
			self._calc_max_single_dim_size(
				self._inherent_dim_df[self.non_grid_dim].to_numpy(),
				1.0
				)
		# convert to (width, height) tuple
		self.inherent_size = \
			tuple(inherent_size_ser[['width','height']].to_list())

	def _calc_max_single_dim_size(self, inherent_sizes, grid_ratios):
		'''
		Calculates the max size along a dimension accounting for grid 
		ratios
		'''
		if np.all(np.isnan(inherent_sizes)):
			max_size = None
		else:
			max_size = np.nanmax(inherent_sizes/grid_ratios)
		return(max_size)
		
	def generate_movie_ims(self, width, height, blank_color):
		'''
		Generates movie images with supplied width and height, and
		returns them in pd dataframe
		'''
		# It's necessary to do this in two loop steps because we don't 
		# know that global_timepoint will be the same for every 
		# movie_maker in self.movie_maker_df, so we need to generate a 
		# dataframe that includes all possible global timepoints first
		####
		# Create a dataframe that will combine image dfs for every 
		# movie_maker in self.movie_maker_df, with indices being 
		# global_timepoint and columns being the indices of 
		# self.movie_maker_df
		self.movie_maker_df.reset_index(inplace = True)
		im_df_list = []
		for idx, row in self.movie_maker_df.iterrows():
			curr_width = int(float(width)*row.width_prop)
			curr_height = int(float(height)*row.height_prop)
			current_im_df = \
				row.movie_maker.generate_movie_ims(
					curr_width, curr_height, blank_color
					)
			current_im_df.columns = [idx]
			im_df_list.append(current_im_df)
		combined_im_df = pd.concat(im_df_list, join = 'outer', axis = 1)
		# fill empty positions (subframes) in combined_im_df with None
		combined_im_df.replace(dict({np.nan:None}),inplace = True)
		# place images on common background at each global timepoint,
		# and add to MovieHolder object
		self.movie_holder = MovieHolder(combined_im_df.index)
		for global_tp, im_row in combined_im_df.iterrows():
			# create background image
			main_im = _create_blank_im(width, height, blank_color)
			# loop through images created by different movie_makers at
			# this global timepoint, placing them on main_im
			# note that indices of row match indices of
			# self.movie_maker_df
			for movie_maker_idx in im_row.index:
				subframe_im = im_row[movie_maker_idx]
				if subframe_im != None:
					curr_width = int(float(width)*\
						self.movie_maker_df.at[movie_maker_idx, 'width_prop'])
					curr_height = int(float(height)*\
						self.movie_maker_df.at[movie_maker_idx, 'height_prop'])
					curr_top_offset = int(float(height)*\
						self.movie_maker_df.at[movie_maker_idx, 'top_pos'])
					curr_left_offset = int(float(width)*\
						self.movie_maker_df.at[movie_maker_idx, 'left_pos'])
					main_im = _place_im(
						subframe_im,
						main_im,
						curr_width,
						curr_height,
						curr_top_offset,
						curr_left_offset
						)
			self.movie_holder.add_im(global_tp, main_im)
		return(self.movie_holder.im_df)

class MovieGenerator(object):
	'''
	Makes movies for individual aspects of colony growth (e.g. colony
	outlines, fluorescence, etc) and combines them into a single frame

	Where colonies in crossphase_colony_id_list are part of the same
	field, they will be included in the movie together

	analysis_config_file is the path to the setup file

	crossphase_colony_id_list is a list of crossphase_colony_ids for
	which movies should be generated

	colony_colors is a set of HEX code strings for colors of every
	colony in crossphase_colony_id_list (in order); if None (default)
	will auto-generate colors using p9 scheme instead
	'''
	def __init__(self,
				crossphase_colony_id_list,
				analysis_config_file = None,
				analysis_config_obj_df = None,
				colony_colors = None
				):
		self.analysis_config_obj_df = \
			check_passed_config(analysis_config_obj_df, analysis_config_file)
		# read in colony properties df
		temp_analysis_config_standin = \
			self.analysis_config_obj_df.iloc[0]['analysis_config']
		comb_colony_prop_df = \
			temp_analysis_config_standin.get_colony_data_tracked_df(
				filter_by_phase = False)
		# add 'global' cross-phase timepoint to col property df
		comb_colony_prop_df_glob_tp = \
			self._add_global_timepoints(comb_colony_prop_df)
		# check that colony ids exist
		if not isinstance(crossphase_colony_id_list,(list,pd.core.series.Series,np.ndarray)):
			raise ValueError('crossphase_colony_id_list must be a list, numpy array, or pandas series')
		if len(crossphase_colony_id_list)==0:
			raise ValueError('crossphase_colony_id_list contains no entries')
		# check that colony ids are unique
		if len(crossphase_colony_id_list) != \
			len(set(crossphase_colony_id_list)):
			raise ValueError(
				'crossphase_colony_id_list contains non-unique colony IDs'
				)
		# add colors
		self.color_df = _generate_colony_colors(
			crossphase_colony_id_list,
			colony_colors
			)
		comb_colony_prop_df_final = \
			pd.merge(left = comb_colony_prop_df_glob_tp, right = self.color_df)
		self._subset_colony_prop_df(comb_colony_prop_df_final,
			crossphase_colony_id_list)

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
				comb_colony_prop_df.phase_num == phase, 'global_timepoint'
				] = \
				comb_colony_prop_df.timepoint[
					comb_colony_prop_df.phase_num == phase
					] + \
				prev_phases_tps
			# update the number of timepoints to add to the next phase
			prev_phases_tps = prev_phases_tps + \
				self.analysis_config_obj_df.at[phase,
					'analysis_config'].max_timepoint_num
		return(comb_colony_prop_df)

	def _subset_colony_prop_df(self, comb_colony_prop_df,
		crossphase_colony_id_list):
		'''
		Creates dict of dataframes containing only colony properties of
		colonies in crossphase_colony_id_list, separated by xy position
		'''
		self.col_prop_df = \
			comb_colony_prop_df[
				comb_colony_prop_df.cross_phase_tracking_id.isin(
					crossphase_colony_id_list)].copy()
		xy_positions = self.col_prop_df.xy_pos_idx.unique()
		if len(xy_positions) > 1:
			raise ValueError('crossphase_colony_id_list must contain colonies '
				'from a single imaging field (xy position)')

	def _parse_fluor_stages(self, mainphase = False, postphase = False):
		'''
		Returns fluor_color_dict, with fluor_channel_column_name keys 
		and fluor_color values for each

		postphase and mainphase are bools; if True, look for 
		fluorescent channels in the respective phase type (main phase 
		and/or post-phase)

		If fluor_color_dict is None (default), assigns channels 
		random-esque colors, trying to maximize overlap sense (green + 
		magenta for 2 channels, magenta + yellow + cyan for 3 channels)
		'''
		# get all fluor channel names specified across 
		fluor_channel_list = []
		max_per_phase_channels = 0
		for phase in self.analysis_config_obj_df.index:
			curr_phase_channel_list = []
			analysis_config = \
				self.analysis_config_obj_df.at[phase, 'analysis_config']
			postphase_analysis_config = \
				self.analysis_config_obj_df.at[
					phase, 'postphase_analysis_config'
					]
			if mainphase:
				fluor_channels = \
					analysis_config.fluor_channel_df.\
						fluor_channel_column_name.to_list()
				curr_phase_channel_list.extend(fluor_channels)
			if postphase and postphase_analysis_config is not None:
				fluor_channels = \
					postphase_analysis_config.fluor_channel_df.\
						fluor_channel_column_name.to_list()
				curr_phase_channel_list.extend(fluor_channels)
			max_per_phase_channels = \
				max(max_per_phase_channels, len(curr_phase_channel_list))
			fluor_channel_list.extend(curr_phase_channel_list)
		fluor_channel_set = set(fluor_channel_list)
		fluor_channel_num = len(fluor_channel_set)
		if fluor_channel_num <= 3:
			color_set = ['cyan', 'magenta', 'yellow']
			fluor_color_dict = \
				dict(zip(fluor_channel_set, color_set[0:fluor_channel_num]))
		else:
			# generate random colors
			fluor_color_df = _generate_colony_colors(
				list(fluor_channel_set),
				randomize_order = False
				)
			fluor_color_dict = \
				dict(zip(
					fluor_color_df.cross_phase_tracking_id.to_list(),
					fluor_color_df.hex_color.to_list()
					))
		return(fluor_color_dict, max_per_phase_channels)

	def make_cell_movie(self, col_shading_alpha, bound_width,
		normalize_intensity = True,
		expansion_pixels = 10, bitdepth = None
		):
		'''
		Sets up movies for main (brightfield or phase contrast) channel
		'''
		movie_maker = _PositionMovieMaker(
			self.col_prop_df,
			col_shading_alpha,
			bound_width,
			normalize_intensity,
			expansion_pixels,
			self.analysis_config_obj_df,
			bitdepth = bitdepth)
		return(movie_maker)

	def make_postfluor_movie(self, col_shading_alpha,
		fluor_channel,
		bound_width,
		normalize_intensity = True,
		fluor_color = 'white',
		expansion_pixels = 10, bitdepth = None):
		'''
		Sets up movies for main (brightfield or phase contrast) channel 
		followed by postphase fluorescence whose channel_label is
		fluor_channel	
		'''
		movie_maker = _PositionMovieMaker(
			self.col_prop_df,
			col_shading_alpha,
			bound_width,
			normalize_intensity,
			expansion_pixels,
			self.analysis_config_obj_df,
			postphase_color = fluor_color,
			postphase_fluor_channel = fluor_channel,
			bitdepth = bitdepth)
		return(movie_maker)

	def make_full_postfluor_movie(self,
		col_shading_alpha,
		bound_width,
		normalize_intensity = True,
		fluor_color_dict = None,
		expansion_pixels = 10,
		bitdepth = None):
		'''
		Sets up movies for main (brightfield or phase contrast) channel 
		followed by postphase fluorescence in all imaged postphase 
		fluorescence channels

		If fluor_color_dict is None (default), assigns channels 
		random-esque colors, trying to maximize overlap sense (green + 
		magenta for 2 channels, magenta + yellow + cyan for 3 channels)
		'''
		if fluor_color_dict is None:
			fluor_color_dict, max_per_phase_channels = \
				self._parse_fluor_stages(postphase = True)
		else:
			_, max_per_phase_channels = \
				self._parse_fluor_stages(postphase = True)
		# calculate proportion of fluorescent channel mixtures
		# good to have the same intensity for a given channel across 
		# phases
		channel_mix_prop = 1.0/float(max_per_phase_channels)
		combined_movie_list = []
		for phase in self.analysis_config_obj_df.index:
			postphase_analysis_config = \
				self.analysis_config_obj_df.at[
					phase, 'postphase_analysis_config'
					]
			if postphase_analysis_config is not None:
				phase_movie_list = []
				fluor_channel_list = \
					postphase_analysis_config.fluor_channel_df.\
						fluor_channel_column_name.to_list()
				for fluor_channel in fluor_channel_list:
					curr_movie = self.make_postfluor_movie(
						col_shading_alpha,
						fluor_channel,
						bound_width,
						normalize_intensity = normalize_intensity,
						fluor_color = fluor_color_dict[fluor_channel],
						expansion_pixels = expansion_pixels,
						bitdepth = bitdepth
						)
					phase_movie_list.append(curr_movie)
				# merge movies from phase with partial intensities
				if len(phase_movie_list)==1:
					phase_movie_maker = phase_movie_list[0]
					combined_movie_list.append(phase_movie_maker)
				elif len(phase_movie_list)>1:
					phase_movie_maker = merge_movie_channels(
						phase_movie_list,
						intens_mult_list=[channel_mix_prop]*len(phase_movie_list)
						)
					combined_movie_list.append(phase_movie_maker)
		# 'merge' movies across phases without rescaling (already scaled) intensities
		if len(combined_movie_list)==1:
			movie_maker_combined = combined_movie_list[0]
		else:
			movie_maker_combined = merge_movie_channels(
				combined_movie_list,
				intens_mult_list=[1]*len(combined_movie_list)
				)
		return(movie_maker_combined)

	def make_fluor_movie(self, fluor_channel, bound_width,
		normalize_intensity = True, fluor_color = 'white',
		expansion_pixels = 10, bitdepth = None):
		'''
		Sets up movies for fluorescent channel whose channel_label is
		fluor_channel
		'''
		# set col_shading_alpha to 0 (don't shade over fluorescent images)
		col_shading_alpha = 0
		movie_maker = _PositionMovieMaker(
			self.col_prop_df,
			col_shading_alpha,
			bound_width,
			normalize_intensity,
			expansion_pixels,
			self.analysis_config_obj_df,
			main_phase_color = fluor_color,
			base_fluor_channel = fluor_channel,
			bitdepth = bitdepth)
		return(movie_maker)

	def make_growth_plot_movie(self,
		facet_override = None, add_growth_line = True, add_lag_line = True):
		'''
		Sets up movies for plot of growth over time
		'''
		movie_maker = _GrowthPlotMovieMaker(
			self.col_prop_df,
			self.color_df,
			self.analysis_config_obj_df,
			facet_override = facet_override,
			add_growth_line = add_growth_line,
			add_lag_line = add_lag_line)
		return(movie_maker)

	def make_property_plot_movie(self,
		plot_prop, facet_override = None, y_label_override = None):
		'''
		Sets up movies for plot of plot_prop over time

		plot_prop can be the name of any column in
		colony_properties_combined file
		'''
		movie_maker = _PlotMovieMaker(
			plot_prop,
			self.col_prop_df,
			self.analysis_config_obj_df,
			facet_override = facet_override,
			y_label_override = y_label_override)
		return(movie_maker)

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
			if randomize_order:
				random.shuffle(hex_col)
	else:
		hex_col = unique_colony_hex_color_list
	color_df = pd.DataFrame({
		'cross_phase_tracking_id': unique_colony_id_list,
		'hex_color': hex_col,
		'rgb_color': [ImageColor.getcolor(c, "RGB") for c in hex_col]
		})
	return(color_df)

def _create_blank_im(width, height, blank_color):
	'''
	Returns blank image of correct color and size
	'''
	required_size = (int(width), int(height))
	blank_im = \
		Image.new('RGB', required_size, blank_color)
	return(blank_im)

def _place_im(subframe_im, main_im, width, height, top_offset, left_offset):
	'''
	Resizes Pillow Image object subframe and places it on Pillow 
	Image object main_im
	'''
	# resize subframe
	required_size = (int(np.round(width)), int(np.round(height)))
	if subframe_im.size != required_size:
		subframe_im = subframe_im.resize(required_size)
	# place subframe on main_im
	offset_position = (int(left_offset), int(top_offset))
	main_im.paste(subframe_im, offset_position)
	return(main_im)

def _ratio_resize_convert(im, width, height, blank_color):
	'''
	Resizes im (which is an Image object) into an image with given 
	width and height, but maintaining original aspect ratio; if 
	the aspect ratio of the desired output image differs from the 
	original, blank space in the output image is filled with 
	blank_color
	'''
	# calculate resizing ratios
	width_ori = im.width
	height_ori = im.height
	width_ratio = float(width)/float(width_ori)
	height_ratio = float(height)/float(height_ori)
	resize_ratio = np.min([width_ratio, height_ratio])
	if np.all(resize_ratio == np.array([width_ratio, height_ratio])):
		# aspect ratio of image is maintained
		if resize_ratio == 1:
			resized_im = im
		else:
			new_width = int(np.round(width*width_ratio))
			new_height = int(np.round(height*height_ratio))
			resized_im = im.resize((new_width, new_height))
	else:
		blank_im = _create_blank_im(width, height, blank_color)
		x_offset = (1-resize_ratio/width_ratio)*float(width)/2
		y_offset = (1-resize_ratio/height_ratio)*float(height)/2
		resized_im = _place_im(
			im,
			blank_im,
			resize_ratio*width_ori,
			resize_ratio*height_ori,
			y_offset,
			x_offset
			)
	return(resized_im)

def _check_rel_ratios(rel_ratio_list, obj_list):
	# Converts ratio_list into a numpy array of relative ratios that 
	# sum to 1
	obj_num = len(obj_list)
	if rel_ratio_list is None:
		rel_ratio_list = np.array([1.0/obj_num]*obj_num)
	else:
		if not isinstance(rel_ratio_list, list) or \
			len(rel_ratio_list) != obj_num:
			raise TypeError('rel_ratio_list must be a list with the same '
				'number of elements as obj_list')
		rel_ratio_list = np.array(rel_ratio_list)
		if not np.all(rel_ratio_list > 0):
			raise ValueError('relative ratios must be greater than 0')
		rel_ratio_list = rel_ratio_list/np.sum(rel_ratio_list)
	return(rel_ratio_list)

def merge_movie_channels(*movie_objects, intens_mult_list = None):
	'''
	Wrapper function for creating an object of _OverlayMovieMaker class

	If intens_mult_list is None, defaults to 1/n intensities for every 
	movie object, where n is the number of objects
	'''
	# make movie_objects into unnested list
	if isinstance(movie_objects[0],list):
		movie_obj_list = list(chain.from_iterable(movie_objects))
	else:
		movie_obj_list = list(movie_objects)
	if intens_mult_list is None:
		obj_num = len(movie_obj_list)
		intens_mult_list = np.array([1.0/obj_num]*obj_num)
	overlay_movie_maker = \
		_OverlayMovieMaker(movie_obj_list, intens_mult_list)
	return(overlay_movie_maker)

def make_movie_grid(*movie_objects, nrow = None, ncol = None, rel_widths = None, rel_heights = None):
	'''
	Creates a grid of movies from movie_objects, which can be either 
	individual movies or other movie grids

	In the case of multiple rows, movies are filled in along each row,
	then moving on to the next row

	nrow is the number of rows movies are arranged in

	ncol (optional) is the number of columns movies are arranged in

	If neither nrow or ncol is passed, make_movie_grid defaults to a 
	single row

	rel_widths is a vector of relative column widths; for example, in 
	a two-column grid, rel_widths = [2, 1] would make the first column 
	twice as wide as the second column. Default is None, in which case 
	column widths are identical

	rel_heights is a vector of relative row heights; works like 
	rel_widths. Default is None, in which case column widths are 
	identical
	'''
	# make movie_objects into unnested list
	if isinstance(movie_objects[0],list):
		movie_obj_list = list(chain.from_iterable(movie_objects))
	else:
		movie_obj_list = list(movie_objects)
	movie_obj_num = len(movie_obj_list)
	# check row and column numbers
	if nrow is None and ncol is None:
		nrow = 1
	if ncol is None:
		ncol = int(movie_obj_num/nrow)
	elif nrow is None:
		nrow = int(movie_obj_num/ncol)
	else:
		expected_obj_num = nrow*ncol
		if movie_obj_num != expected_obj_num:
			raise ValueError(
				'Number of movie objects is '+str(movie_obj_num)+
				', but expected number is nrow*ncol, '+str(expected_obj_num)
				)
	if not isinstance(nrow, int):
		raise TypeError('nrow must be an integer')
	if not isinstance(ncol, int):
		raise TypeError('ncol must be an integer')
	# create movie grids for every row, then combine them into a 
	# single grid
	row_grid_list = []
	for curr_row in range(0, nrow):
		start_idx = curr_row*ncol
		end_idx = start_idx+ncol
		curr_movie_obj_list = movie_obj_list[start_idx:end_idx]
		row_grid = _MovieGrid(curr_movie_obj_list, 'row', rel_widths)
		row_grid_list.append(row_grid)
	# combine row_grids
	full_grid = _MovieGrid(row_grid_list, 'column', rel_heights)
	return(full_grid)

def save_movie(movie_saver_obj,
		movie_output_path, movie_name, movie_format,
		movie_width = None, movie_height = None,
		blank_color = 'white', duration = 1000, loop = 0,
		jpeg_quality = 95):
	'''
	Saves movie
	'''
	movie_saver_obj.write_movie(
		movie_format, movie_output_path, movie_name,
		movie_width, movie_height,
		blank_color, duration, loop, jpeg_quality)

def make_position_movie(
	xy_pos_idx,
	analysis_config_file = None,
	analysis_config_obj_df = None,
	colony_subset = 'growing',
	movie_format = 'gif'
	):
	'''
	Creates a movie for a single imaging field numbered xy_pos_idx, 
	showing the colony outlines on the background of the main imaging 
	channel, and the change in log area over time

	xy_pos_idx is the integer of the position for which the movie 
	should be made

	Either analysis_config_file or an existing analysis_config_obj_df 
	must be passed

	colony_subset can be:
	-	'growing': to label only those colonies that receive a growth 
		rate measurement after filtration
	-	'tracked': to label all colonies that were tracked, but not 
		include those that were recognized but then removed because
		they were e.g. a minor part of a broken-up colony
	-	'all': to label all colonies detected by PIE
	(default is 'growing')

	movie_format can be:
	-	'jpeg'/'jpg' (creates a folder with a .jpg format image for
		each timepoint)
	-	'tiff'/'tif' (creates a folder with a .tif format image for
		each timepoint)
	-	'gif' (default; creates gif)
	-	'h264' (video codec; creates movie with .mov extension)
	-	'mjpg'/'mjpeg' (video codec; creates movie with .mov extension)
	'''
	# check that only analysis_config_obj_df or
	# analysis_config_file is passed, and get analysis_config_obj_df
	analysis_config_obj_df = check_passed_config(
		analysis_config_obj_df, analysis_config_file
		)
	# read in colony properties df
	temp_analysis_config_standin = \
		analysis_config_obj_df.iloc[0]['analysis_config']
	if colony_subset == 'growing':
		colony_df = temp_analysis_config_standin.get_gr_data()
	elif colony_subset == 'tracked':
		colony_df = \
			temp_analysis_config_standin.get_colony_data_tracked_df(
				remove_untracked = True, filter_by_phase = False)
	elif colony_subset == 'all':
		colony_df = \
			temp_analysis_config_standin.get_colony_data_tracked_df(
				remove_untracked = False, filter_by_phase = False)
	crossphase_colony_id_list = \
		list(
			colony_df[
				colony_df.xy_pos_idx == xy_pos_idx
				].cross_phase_tracking_id.unique()
			)
	if crossphase_colony_id_list:
		# generate movies
		movie_generator = MovieGenerator(
			crossphase_colony_id_list,
			analysis_config_obj_df = analysis_config_obj_df,
			)
		col_shading_alpha = 0.5
		normalize_intensity = True
		bound_width = 2
		# set expansion pixels to ridiculously large number so code 
		# defaults to showing full frame
		expansion_pixels = 10**10
		# if any post-phase fluor stages exist, default to making
		# post-fluor movie
		if any(
			[pp_analysis_config is not None for pp_analysis_config in 
			analysis_config_obj_df.postphase_analysis_config.to_list()]
			):
			cell_movie = \
				movie_generator.make_full_postfluor_movie(
					col_shading_alpha,
					bound_width,
					normalize_intensity=normalize_intensity,
					)
		else:
			cell_movie = \
				movie_generator.make_cell_movie(
					col_shading_alpha,
					bound_width,
					normalize_intensity,
					expansion_pixels = 10**10)
		gr_plot = \
			movie_generator.make_growth_plot_movie()
		# write movie (try to ensure plot is a square for prettyness)
		cells_hw_ratio = float(cell_movie.inherent_size[1])/float(cell_movie.inherent_size[1])
		movie_grid = \
			make_movie_grid(cell_movie, gr_plot, rel_widths = [1, cells_hw_ratio])
		movie_output_path = temp_analysis_config_standin.movie_folder
		movie_name = 'xy'+str(xy_pos_idx)+'_'+colony_subset+'_colonies_movie'
		save_movie(movie_grid, movie_output_path, movie_name, movie_format)

